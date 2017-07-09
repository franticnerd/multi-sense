import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np

class RelationDataset(Dataset):

    def __init__(self, data_file, multi_sense=False, n_sense = 1):
        self.instances = pd.read_table(data_file, header=None)
        self.multi_sense = multi_sense
        self.n_sense = n_sense
        # self.get_negative_samples()

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
        print weights
        for idx in counter:
            count = counter[idx]
            weights[idx] = count
        print weights
        print sum(weights)
        weights = weights / sum(weights)
        print weights


    # get negative samples for y
    def sample_negatives(self):
        pass



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
        return len(self.id_to_description) * self.n_sense

    def get_description(self, id):
        return self.id_to_description[id / self.n_sense]


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


