import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import numpy.random as nprd
import math


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
        with open(data_file, 'r') as fin:
            for line in fin:
                items = line.strip().split('\t')
                idx = int(items[0])
                description = items[1]
                self.id_to_description[idx] = description
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


def load_data(data_file, multi_sense=False, n_sense=1):
    data = RelationDataset(data_file, multi_sense, n_sense)
    return data


# start = time.time()
# dataloader = DataLoader(data, batch_size=4, shuffle=False, num_workers=2)
# for i_batch, sample_batched in enumerate(dataloader):
#     if i_batch == 2:
#         print str(sample_batched)
#         print len(sample_batched)
# end = time.time()
# print end - start


