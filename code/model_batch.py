import time
from random import randint

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class ReconBatch(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_vocab_size):
        super(ReconBatch, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.linear = nn.Linear(embedding_dim, output_vocab_size)
    def forward(self, inputs, length_weights):
        embeds = self.embedding(inputs)
        out = torch.sum(embeds, dim=1)
        out = torch.bmm(length_weights, out).squeeze(1)
        out = self.linear(out)
        log_probs = F.log_softmax(out)
        return log_probs


class RelationDataset(Dataset):
    def __init__(self, data_file, multi_sense=False, n_sense = 1):
        self.instances = pd.read_table(data_file, header=None)
        self.multi_sense = multi_sense
        self.n_sense = n_sense if self.multi_sense else 1
        self.max_len = self.get_max_len()
    def get_max_len(self):
        max_len = 0
        for idx in xrange(len(self.instances)):
            features = self.instances.ix[idx, 1].strip().split()
            if len(features) > max_len:
                max_len = len(features)
        return max_len
    def __len__(self):
        return len(self.instances)
    def __getitem__(self, idx):
        label = [self.instances.ix[idx, 0]]
        input = map(int, str(self.instances.ix[idx, 1]).strip().split())
        length = len(input)
        features = np.zeros(self.max_len * self.n_sense, dtype=np.int)
        for i in xrange(length):
            for j in xrange(self.n_sense):
                pos = i * self.n_sense + j
                # need to plus 1 because 0 is preserved as padding idx
                value = input[i] * self.n_sense + j + 1
                features[pos] = value
        return torch.from_numpy(features), torch.LongTensor([length]), torch.LongTensor(label)



class Vocab():
    def __init__(self, data_file, multi_sense=False, n_sense=1, offset=0):
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
        self.offset = offset

    def size(self):
        return len(self.id_to_description) * self.n_sense + self.offset

    # get the description for a converted id
    def get_description(self, idx):
        raw_id = (idx - self.offset) / self.n_sense
        return self.id_to_description[raw_id]

    # get the converted id for a given description
    def get_id(self, description):
        raw_id = self.description_to_id[description]
        return [raw_id * self.n_sense + i + self.offset for i in xrange(self.n_sense)]


# check whether the ground truth appears in the top-K list, for computing the hit ratio
def get_num_correct(ground_truth, output):
    correct = 0
    _, predicted = torch.max(output, 1)
    correct += (predicted == ground_truth).sum()
    return correct


def get_groundtruth_rank_full(ground_truth, output):
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
def get_groundtruth_rank_pool(ground_truth, output, num_cand=10):
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


# train_file = '/Users/chao/data/projects/multi-sense-embedding/toy/input/train.txt'
# test_file = '/Users/chao/data/projects/multi-sense-embedding/toy/input/test.txt'
# x_vocab_file = '/Users/chao/data/projects/multi-sense-embedding/toy/input/words.txt'
# y_vocab_file = '/Users/chao/data/projects/multi-sense-embedding/toy/input/locations.txt'

train_file = '/Users/chao/data/projects/multi-sense-embedding/tweets-10k/input/train.txt'
test_file = '/Users/chao/data/projects/multi-sense-embedding/tweets-10k/input/test.txt'
x_vocab_file = '/Users/chao/data/projects/multi-sense-embedding/tweets-10k/input/words.txt'
y_vocab_file = '/Users/chao/data/projects/multi-sense-embedding/tweets-10k/input/locations.txt'

embedding_dim = 5

train_data = RelationDataset(train_file, False, 1)
test_data = RelationDataset(test_file, False, 1)
x_vocab = Vocab(x_vocab_file, False, 1, offset=1)
y_vocab = Vocab(y_vocab_file, False, 1)

dataloader = DataLoader(train_data, batch_size=10, shuffle=False, num_workers=1)
test_dataloader = DataLoader(test_data, batch_size=10, shuffle=False, num_workers=1)

# for i_batch, sample_batched in enumerate(dataloader):
#     print i_batch, len(sample_batched)
#     print sample_batched[0]
#     print sample_batched[1]
#     print sample_batched[2]
    # print sample_batched[1]

pre_time, forward_time, backward_time = 0, 0, 0
model = ReconBatch(x_vocab.size(), embedding_dim, y_vocab.size())
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
for epoch in xrange(50):
    for data_batch in dataloader:
        p_start_time = time.time()
        features = Variable(data_batch[0])
        length_weights = Variable(1.0/data_batch[1].float()).view(-1,1,1)
        labels = Variable(data_batch[2]).view(-1)
        p_end_time = time.time()
        pre_time += p_end_time - p_start_time

        f_start_time = time.time()
        outputs = model(features, length_weights)
        f_end_time = time.time()
        forward_time += f_end_time - f_start_time
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        b_start_time = time.time()
        loss.backward()
        optimizer.step()
        b_end_time = time.time()
        backward_time += (b_end_time - b_start_time)
print 'time', pre_time, forward_time, backward_time



num_correct, ranks, pool_ranks = 0, [], []
for data_batch in test_dataloader:
    features = Variable(data_batch[0])
    length_weights = Variable(1.0/data_batch[1].float()).view(-1,1,1)
    ground_truth = data_batch[2].view(-1, 1)
    output = model(features, length_weights)
    num_correct += get_num_correct(ground_truth, output.data)
    ranks.extend(get_groundtruth_rank_full(ground_truth, output.data))
    pool_ranks.extend(get_groundtruth_rank_pool(ground_truth, output.data))
accuracy = float(num_correct) / float(len(test_data))
mr_full = np.mean(ranks)
mrr_full = np.mean([1.0/r for r in ranks])
mr_pool = np.mean(pool_ranks)
mrr_pool = np.mean([1.0/r for r in pool_ranks])
print 'performance', accuracy, mr_full, mrr_full, mr_pool, mrr_pool

