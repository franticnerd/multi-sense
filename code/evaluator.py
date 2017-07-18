from random import randint

from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from torch.autograd import Variable

from dataset import ClassifyDataSet
from utils import format_list_to_string, ensure_directory_exist
from utils import read_first_line


class Evaluator:

    def __init__(self, opt):
        self.opt = opt
        self.n_sense = opt['n_sense']
        self.dim = opt['embedding_dim']
        self.n_epoch = opt['n_epoch']
        self.dataset = opt['data_dir'].split('/')[-2]
        self.batch_size = opt['batch_size']
        self.learning_rate = opt['learning_rate']

    # eval only accuracy
    def eval_accuracy(self, model, test_data):
        num_correct = 0
        for data_batch in test_data:
            features, length_weights, word_masks, ground_truth = self.convert_to_variable(data_batch)
            output = model(features, length_weights, word_masks)
            num_correct += self.get_num_correct(ground_truth, output.data)
        accuracy = float(num_correct) / float(len(test_data) * self.batch_size)
        return accuracy

    # eval accuracy and mrr
    def eval(self, model, test_data):
        num_correct, ranks, pool_ranks = 0, [], []
        for data_batch in test_data:
            features, length_weights, word_masks, ground_truth = self.convert_to_variable(data_batch)
            output = model(features, length_weights, word_masks)
            num_correct += self.get_num_correct(ground_truth, output.data)
            ranks.extend(self.get_groundtruth_rank_full(ground_truth, output.data))
            pool_ranks.extend(self.get_groundtruth_rank_pool(ground_truth, output.data))
        accuracy = float(num_correct) / float(len(test_data) * self.batch_size)
        mr_full = np.mean(ranks)
        mrr_full = np.mean([1.0/r for r in ranks])
        mr_pool = np.mean(pool_ranks)
        mrr_pool = np.mean([1.0/r for r in pool_ranks])
        return accuracy, mr_full, mrr_full, mr_pool, mrr_pool

    def convert_to_variable(self, data_batch):
        features = Variable(data_batch[0])
        length_weights = Variable(1.0/data_batch[1].float()).view(-1, 1, 1)
        word_masks = data_batch[2]
        ground_truth = data_batch[3].view(-1, 1)
        return features, length_weights, word_masks, ground_truth




    # check whether the ground truth appears in the top-K list, for computing the hit ratio
    def get_num_correct(self, ground_truth, output):
        correct = 0
        _, predicted = torch.max(output, 1)
        correct += (predicted == ground_truth).sum()
        return correct

    def get_groundtruth_rank_full(self, ground_truth, output):
        ret = []
        for true_id, predicted in zip(ground_truth, output):
            true_id = true_id[0]
            true_id_score = predicted[true_id]
            rank = 1
            for score in predicted:
                if score > true_id_score:
                    rank += 1
            ret.append(rank)
        return ret

    # get the groundtruth rank from a pool of candidates
    def get_groundtruth_rank_pool(self, ground_truth, output, num_cand=10):
        ret = []
        for true_id, predicted in zip(ground_truth, output):
            true_id = true_id[0]
            true_id_score = predicted[true_id]
            rand_idx_set = set([true_id])
            while len(rand_idx_set) < num_cand:
                r = randint(0, len(predicted)-1)
                rand_idx_set.add(r)
            rank = 1
            for r in rand_idx_set:
                if predicted[r] > true_id_score:
                    rank += 1
            ret.append(rank)
        return ret

    def write_performance(self, model_type, metrics, train_time):
        def get_header_string():
            header = ['acc', 'mr_f', 'mrr_f', 'mr_p', 'mrr_p',
                      'S', 'D', 'B', 'E', 'l_rate',
                      'data_set', 't_sec', 'model_type']
            return format_list_to_string(header, '\t')
        def get_perf_string(metrics, train_time):
            content = [metrics[0], metrics[1], metrics[2], metrics[3], metrics[4],
                       self.n_sense, self.dim, self.batch_size, self.n_epoch, self.learning_rate,
                       self.dataset, train_time, model_type]
            return format_list_to_string(content, '\t')
        header_string = get_header_string()
        perf_string = get_perf_string(metrics, train_time)
        print header_string + '\n' + perf_string
        perf_file = self.opt['performance_file']
        ensure_directory_exist(perf_file)
        file_header = read_first_line(perf_file)
        with open(perf_file, 'a') as fout:
            if file_header != header_string:
                fout.write(header_string + '\n')
            fout.write(perf_string + '\n')


    def eval_classification(self, opt, model_type, model):
        data = ClassifyDataSet(opt, model_type, model)
        features_train = data.features_train
        features_test = data.features_test
        labels_train = data.labels_train
        labels_test = data.labels_test

        standard_scaler = StandardScaler()
        features_train = standard_scaler.fit_transform(features_train)
        features_test = standard_scaler.transform(features_test)
        model = self.train_classifier(features_train, labels_train)
        print self.eval_classifier(model, features_test, labels_test)

    def train_classifier(self, features, labels):
        model = LogisticRegression()
        model.fit(features, labels)
        return model

    def eval_classifier(self, model, features, labels):
        expected = labels
        predicted = model.predict(features)
        prfs = precision_recall_fscore_support(expected, predicted)
        precision, recall, f1 = prfs[0][1], prfs[1][1], prfs[2][1]
        return precision, recall, f1



class CaseEvaluator:

    def __init__(self, model, dataset, opt):
        self.model = model
        self.data = dataset
        self.case_seed_file = opt['case_seed_file']
        self.case_output_file = opt['case_output_file']
        self.K = opt['K']

    def run_case_study(self):
        case_seeds = self.load_case_seeds()
        for case in case_seeds:
            # it is a list because there can be multiple senses
            idx_list = self.data.x_vocab.get_feature_ids(case)
            for idx in idx_list:
                scores, neighbor_idx = self.find_topk(idx)
                self.write_one_case(idx, neighbor_idx, scores)

    # find the top k similar units to the query idx
    def find_topk(self, idx):
        embedder = self.model.embedding
        query_vec = embedder.weight[idx]
        similarities = self.calc_similarities(embedder, query_vec)
        scores, neighbor_idx = torch.topk(similarities, self.K)
        return scores.squeeze().data.tolist(), neighbor_idx.squeeze().data.tolist()

    def calc_similarities(self, embedder, query_vec):
        similarities = torch.mm(query_vec.view(1,-1), embedder.weight.transpose(0, 1))
        return similarities
        # ret = []
        # for i in xrange(self.embedding.weight.size()[0]):
        #     embedding = self.embedding.weight[i]
        #     norm_prod = embedding.norm() * query_embedding.norm()
        #     denominator = np.max([1e-8, norm_prod.data[0]])
        #     similarity = np.dot(query_embedding.data.tolist(), embedding.data.tolist()) / denominator
        #     ret.append(similarity)
        # return Variable(torch.Tensor(ret)).view(1, -1)

    def load_case_seeds(self):
        ret = []
        with open(self.case_seed_file, 'r') as fin:
            for line in fin:
                ret.append(line.strip())
        return ret

    def write_one_case(self, idx, neighbor_ids, scores):
        neighbor_info = [idx]
        description = self.data.x_vocab.get_description(idx)
        neighbor_info.append(description)
        for (neighbor_idx, score) in zip(neighbor_ids, scores):
            description = self.data.x_vocab.get_description(neighbor_idx)
            neighbor_info.append([description, score])
        neighbor_info = format_list_to_string(neighbor_info, '\t')
        # print neighbor_info
        with open(self.case_output_file, 'a') as fout:
            fout.write(neighbor_info + '\n')

