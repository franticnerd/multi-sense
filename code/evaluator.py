from random import randint

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
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
        self.regu_strength = opt['regu_strength']
        self.dropout = opt['dropout']
        self.perf_file = opt['data_dir'] + 'output/performance.txt'
        self.instance_analysis_path = opt['data_dir'] + 'instance/'
        ensure_directory_exist(self.perf_file)
        self.test_data_file = opt['data_dir'] + 'input/test.txt'
        self.x_vocab_file = opt['data_dir'] + 'input/words.txt'
        self.criterion = nn.NLLLoss()

    # eval only accuracy
    def eval_accuracy_and_loss(self, model, test_data):
        model.eval()
        num_correct, avg_loss, test_size = 0, 0, 0
        for data_batch in test_data:
            features, length_weights, word_masks, ground_truth = self.convert_to_variable(data_batch)
            output = model(features, length_weights, word_masks)
            num_correct += self.get_num_correct(ground_truth, output.data)
            loss = self.criterion(output, Variable(ground_truth).view(-1))
            avg_loss += loss.data[0]
            test_size += ground_truth.size()[0]
        accuracy = float(num_correct) / float(test_size)
        avg_loss /= float(test_size)
        return accuracy, avg_loss

    # eval accuracy and mrr
    def eval(self, model, model_type, test_data):
        model.eval()
        avg_loss, test_size, num_correct, ranks, pool_ranks, correct_idx = 0, 0, 0, [], [], []
        for data_batch in test_data:
            features, length_weights, word_masks, ground_truth = self.convert_to_variable(data_batch)
            output = model(features, length_weights, word_masks)
            num_correct += self.get_num_correct(ground_truth, output.data)
            batch_correct = self.get_correct_idx(ground_truth, output.data)
            correct_idx.extend(batch_correct)
            ranks.extend(self.get_groundtruth_rank_full(ground_truth, output.data))
            pool_ranks.extend(self.get_groundtruth_rank_pool(ground_truth, output.data))
            output = model(features, length_weights, word_masks)
            loss = self.criterion(output, Variable(ground_truth).view(-1))
            avg_loss += loss.data[0]
            test_size += ground_truth.size()[0]
        accuracy = float(num_correct) / float(test_size)
        mr_full = np.mean(ranks)
        mrr_full = np.mean([1.0/r for r in ranks])
        mr_pool = np.mean(pool_ranks)
        mrr_pool = np.mean([1.0/r for r in pool_ranks])
        avg_loss /= float(test_size)
        precision, recall, f1 = self.eval_classification(model, model_type)
        return accuracy, avg_loss, mr_full, mrr_full, mr_pool, mrr_pool, precision, recall, f1, correct_idx

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

    # check whether the ground truth appears in the top-K list, for computing the hit ratio
    def get_correct_idx(self, ground_truth, output):
        _, predicted = torch.max(output, 1)
        return (predicted == ground_truth).view(-1).tolist()

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
            header = ['acc', 'mr_f', 'mrr_f', 'mr_p', 'mrr_p', 'pre', 'rec', 'f1',
                      'S', 'D', 'B', 'E', 'l_rate',
                      'data_set', 't_sec', 'model_type']
            return format_list_to_string(header, '\t')
        def get_perf_string(metrics, train_time):
            content = []
            content.extend(metrics[:-1])
            content.extend([self.n_sense, self.dim, self.batch_size, self.n_epoch, self.learning_rate,\
                            self.regu_strength, self.dropout, self.dataset, train_time, model_type])
            return format_list_to_string(content, '\t')
        # quantitative analysis
        header_string = get_header_string()
        perf_string = get_perf_string(metrics, train_time)
        print header_string + '\n' + perf_string
        file_header = read_first_line(self.perf_file)
        with open(self.perf_file, 'a') as fout:
            if file_header != header_string:
                fout.write(header_string + '\n')
            fout.write(perf_string + '\n')
        # error analysis
        error_indicator = metrics[-1]
        error_indicator_file = self.instance_analysis_path + str(model_type) + '.txt'
        ensure_directory_exist(error_indicator_file)
        with open(error_indicator_file, 'w') as fout:
            for element in error_indicator:
                fout.write(str(element) + '\n')

    def eval_classification(self, model, model_type):
        try:
            data = ClassifyDataSet(self.opt, model_type, model)
        except:
            print 'Cannot load classification data set.'
            return 0, 0, 0
        features_train = data.features_train
        features_test = data.features_test
        labels_train = data.labels_train
        labels_test = data.labels_test

        standard_scaler = StandardScaler()
        features_train = standard_scaler.fit_transform(features_train)
        features_test = standard_scaler.transform(features_test)
        model = self.train_classifier(features_train, labels_train)
        return self.eval_classifier(model, features_test, labels_test)

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

