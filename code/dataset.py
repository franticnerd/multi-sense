import math
from collections import Counter

import torch
import numpy as np
import numpy.random as nprd
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class RelationDataset(Dataset):

    def __init__(self, data_file, n_sense=1, feature_id_offset=0):
        self.instances = pd.read_table(data_file, header=None)
        self.n_sense = n_sense
        # the offset of the idx of the features, by default preserve 0 as the dumb embedding
        self.feature_id_offset = feature_id_offset
        # the maximum number of non-zero features in an instance
        self.max_feature_len = self.get_max_feature_len()

    def get_max_feature_len(self):
        max_len = 0
        for idx in xrange(len(self.instances)):
            features = self.instances.ix[idx, 1].strip().split()
            if len(features) > max_len:
                max_len = len(features)
        return max_len

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        label = self.instances.ix[idx, 0]
        feature_input = map(int, str(self.instances.ix[idx, 1]).strip().split())
        feature_length = len(feature_input)
        # every instance has faeture length as self.max_len * self.n_sense, padded with zeros by default
        features = np.zeros(self.max_feature_len * self.n_sense, dtype=np.int)
        word_mask = np.ones(self.max_feature_len, dtype=np.int)
        for i in xrange(feature_length):
            for j in xrange(self.n_sense):
                pos = i * self.n_sense + j
                # need to plus the offset (1) because 0 is preserved as padding idx
                value = self.feature_id_offset + feature_input[i] * self.n_sense + j
                features[pos] = value
            word_mask[i] = 0
        return torch.from_numpy(features), torch.LongTensor([feature_length]), torch.from_numpy(word_mask).byte(), torch.LongTensor([label])

    # TODO: need to modify the following functions to account for offset
    # get the weights for sampling from multinomial distributions, used for negative sampling models
    def gen_multinomial_dist(self, n_label):
        weights = np.zeros(n_label)
        y_labels = self.instances.ix[:, 0].tolist()
        counter = Counter(y_labels)
        for idx in counter:
            count = counter[idx]
            weights[idx] = math.pow(count, 0.75)
        self.weights = weights / sum(weights)

    # get negative samples for y, used for negative sampling models
    def sample_negatives(self, n_sample, tgt_idx):
        n_label = len(self.weights)
        ret = []
        while len(ret) < n_sample:
            rand_idx = nprd.choice(n_label, p=self.weights)
            if rand_idx != tgt_idx:
                ret.append(rand_idx)
        return ret


class Vocab:
    def __init__(self, data_file, n_sense=1, id_offset=0):
        self.rawid_to_description = {}
        self.description_to_rawid = {}
        with open(data_file, 'r') as fin:
            for line in fin:
                items = line.strip().split('\t')
                raw_id = int(items[0])
                description = items[1]
                self.rawid_to_description[raw_id] = description
                self.description_to_rawid[description] = raw_id
        self.n_sense = n_sense
        self.id_offset = id_offset

    # the size of the vocabulory, namely the number of converted feature ids (including 0)
    def size(self):
        return len(self.rawid_to_description) * self.n_sense + self.id_offset

    # get the description for a converted id
    def get_description(self, idx):
        raw_id = (idx - self.id_offset) / self.n_sense
        return self.rawid_to_description[raw_id]

    # get the feature id list for a description
    def get_feature_ids(self, description):
        raw_id = self.description_to_rawid[description]
        return [raw_id * self.n_sense + i + self.id_offset for i in xrange(self.n_sense)]


class DataSet:
    def __init__(self, opt, model_type):
        n_sense = 1 if model_type in ['recon', 'attn'] else opt['n_sense']
        x_vocab_file = opt['x_vocab_file']
        y_vocab_file = opt['y_vocab_file']
        train_file = opt['train_data_file']
        valid_file = opt['valid_data_file']
        test_file = opt['test_data_file']
        batch_size = opt['batch_size']
        n_worker = opt['data_worker']
        # load the data
        self.x_vocab = Vocab(x_vocab_file, n_sense, id_offset=1)
        self.y_vocab = Vocab(y_vocab_file, 1)
        # self.y_vocab = Vocab(y_vocab_file, n_sense)
        train_data = RelationDataset(train_file, n_sense, feature_id_offset=1)
        valid_data = RelationDataset(valid_file, n_sense, feature_id_offset=1)
        test_data = RelationDataset(test_file, n_sense, feature_id_offset=1)
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=n_worker)
        self.valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=n_worker)
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=n_worker)

        # # for debug mode
        # for batched in self.train_loader:
        #     print batched[0]
        #     print batched[1]
        #     print batched[2]

