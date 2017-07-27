import torch
from utils import format_list_to_string, calc_cosine, calc_sim, find_topk_neighbors, ensure_directory_exist


class CaseAnalyzer:

    def __init__(self, model, dataset, opt):
        self.model = model
        self.vocab = dataset.x_vocab
        self.n_sense = opt['n_sense']
        self.emb_matrix = model.embedding.weight.data
        self.n_words = (self.emb_matrix.size()[0] - 1) / self.n_sense
        self.sim_thre = 1e-3
        self.opt = opt
        self.K = 20
        case_dir = opt['data_dir'] + 'case/'
        self.dup_sense_file = case_dir + 'duplicate_sense_words.txt'
        self.case_seed_file = case_dir + 'query_words.txt'
        self.case_output_file = case_dir + 'similar_words.txt'

    # find the words whose senses are duplicate
    def find_duplicate_sense_words(self):
        if self.n_sense != 2:
            print 'Similar sense detection is only supported for n_sense = 2.'
            return
        print 'Start detecting duplicate sense for %s words.' % self.n_words
        words = []
        for i in xrange(self.n_words):
            idx = i * self.n_sense + 1
            vec_a = self.emb_matrix[idx]
            vec_b = self.emb_matrix[idx + 1]
            if self.is_similar(vec_a, vec_b, mode='norm'):
                word = self.vocab.get_description(idx)
                words.append(word)
        ensure_directory_exist(self.dup_sense_file)
        with open(self.dup_sense_file, 'w') as fout:
            for w in words:
                fout.write(w + '\n')

    def is_similar(self, vec_a, vec_b, mode='cos'):
        if mode == 'cos':
            similarity = calc_cosine(vec_a, vec_b)
            return True if similarity >= 1.0 - self.sim_thre else False
        elif mode == 'norm':
            similarity = torch.norm(vec_a - vec_b)
            return True if similarity <= self.sim_thre else False
        else:
            print 'Similarity computation must be either cos or norm.'
            return False

    # find top-K words for the given seeds
    def find_similar_words(self):
        try:
            query_words = self.load_query_words()
            for query in query_words:
                # it is a list because there can be multiple senses
                idx_list = self.vocab.get_feature_ids(query)
                for idx in idx_list:
                    query_vec = self.emb_matrix[idx]
                    similarities = self.get_similarities(query_vec)
                    scores, neighbor_idx = find_topk_neighbors(similarities, self.K)
                    self.write_one_case(idx, neighbor_idx, scores)
        except:
            print 'Case study failed. Check whether the given cases are valid.'
            return

    def load_query_words(self):
        ret = []
        with open(self.case_seed_file, 'r') as fin:
            for line in fin:
                ret.append(line.strip())
        return ret

    # find the similarity between the query vector and all the other vectors
    def get_similarities(self, query_vec):
        similarities = []
        for i in xrange(self.emb_matrix.size()[0]):
            embedding = self.emb_matrix[i]
            similarity = calc_sim(query_vec, embedding, mode='dot')
            similarities.append(similarity)
        return torch.FloatTensor(similarities)

    def write_one_case(self, idx, neighbor_ids, scores):
        neighbor_info = [idx]
        description = self.vocab.get_description(idx)
        neighbor_info.append(description)
        for (neighbor_idx, score) in zip(neighbor_ids, scores):
            description = self.vocab.get_description(neighbor_idx)
            neighbor_info.append([description, score])
        neighbor_info = format_list_to_string(neighbor_info, ' ')
        # print neighbor_info
        ensure_directory_exist(self.case_output_file)
        with open(self.case_output_file, 'a') as fp:
            fp.write(neighbor_info + '\n')
