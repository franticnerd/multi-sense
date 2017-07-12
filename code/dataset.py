import math
from collections import Counter

import numpy as np
import numpy.random as nprd
import pandas as pd
from torch.utils.data import Dataset


class RelationDataset(Dataset):

    def __init__(self, data_file, multi_sense=False, n_sense = 1):
        self.instances = pd.read_table(data_file, header=None)
        self.multi_sense = multi_sense
        self.n_sense = n_sense

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        label = [self.instances.ix[idx, 0]]
        input = map(int, str(self.instances.ix[idx, 1]).strip().split())
        if self.multi_sense:
            input = [e * self.n_sense + i for e in input for i in xrange(self.n_sense)]
        return input, label

    # get the weights for sampling from multinomial distributions
    def gen_multinomial_dist(self, n_label):
        weights = np.zeros(n_label)
        y_labels = self.instances.ix[:, 0].tolist()
        counter = Counter(y_labels)
        for idx in counter:
            count = counter[idx]
            weights[idx] = math.pow(count, 0.75)
        self.weights = weights / sum(weights)


    # get negative samples for y
    def sample_negatives(self, n_sample, tgt_idx):
        n_label = len(self.weights)
        ret = []
        while len(ret) < n_sample:
            rand_idx = nprd.choice(n_label, p=self.weights)
            if rand_idx != tgt_idx:
                ret.append(rand_idx)
        return ret


class Vocab():

    def __init__(self, data_file, multi_sense=False, n_sense=1):
        self.id_to_description = {}
        self.description_to_id = {}
        with open(data_file, 'r') as fin:
            for line in fin:
                items = line.strip().split('\t')
                idx = int(items[0])
                description = items[1]
                self.id_to_description[idx] = description
                self.description_to_id[description] = idx
        self.multi_sense = multi_sense
        self.n_sense = n_sense


    def size(self):
        if self.multi_sense:
            return len(self.id_to_description) * self.n_sense
        else:
            return len(self.id_to_description)


    def get_description(self, id):
        if self.multi_sense:
            return self.id_to_description[id / self.n_sense]
        else:
            return self.id_to_description[id]

    def get_id(self, description):
        idx = self.description_to_id[description]
        if self.multi_sense:
            return [idx * self.n_sense + i for i in xrange(self.n_sense)]
        else:
            return [idx]


class DataSet:

    def __init__(self, opt, model_type):
        self.opt = opt
        self.model_type = model_type
        # set the sense parameters based on model type
        if model_type in ['recon', 'attn']:
            self.multi_sense = False
            self.n_sense = 1
        else:
            self.multi_sense = True
            self.n_sense = self.opt['n_sense']
        self.x_vocab_file = opt['x_vocab_file']
        self.y_vocab_file = opt['y_vocab_file']
        self.train_file = opt['train_data_file']
        self.test_file = opt['test_data_file']
        self.valid_file = opt['valid_data_file']
        self.load_data()

    def load_data(self):
        self.x_vocab = Vocab(self.x_vocab_file, self.multi_sense, self.n_sense)
        # self.y_vocab = Vocab(y_vocab_file, self.multi_sense, self.n_sense)
        self.y_vocab = Vocab(self.y_vocab_file, False, 1)
        self.train_data = RelationDataset(self.train_file, self.multi_sense, self.n_sense)
        self.test_data = RelationDataset(self.test_file, self.multi_sense, self.n_sense)
        self.valid_data = RelationDataset(self.valid_file, self.multi_sense, self.n_sense)


    # # def get_x_vocab_size(self):
    # #     return self.x_vocab.size()
    #
    # def get_y_vocab_size(self):
    #     return self.y_vocab.size()

        # # y_vocab = Vocab(pd['y_vocab_file'], multi_sense, n_sense)
        # y_vocab = Vocab(pd['y_vocab_file'], False, 1)
        # train_data = RelationDataset(pd['train_data_file'], multi_sense, n_sense)
        # test_data = RelationDataset(pd['test_data_file'], multi_sense, n_sense)
        # validation_data = RelationDataset(pd['valid_data_file'], multi_sense, n_sense)
        # return x_vocab, y_vocab, train_data, test_data, validation_data
