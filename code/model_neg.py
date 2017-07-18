from random import randint
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset_neg import RelationDataset, Vocab
from loss import NSNLLLoss
from models import constants

# data_dir = '../data/tweets-10k/'
data_dir = '../data/tweets-10k/'
train_file = data_dir + 'input/train.txt'
test_file = data_dir + 'input/test.txt'
valid_file = data_dir + 'input/test.txt'
x_vocab_file = data_dir + 'input/words.txt'
y_vocab_file = data_dir + 'input/locations.txt'
train_log_file = data_dir + 'output/train_log.txt'
performance_file = data_dir + 'output/performance.txt'
model_path = data_dir + 'model/'

n_sense = 2
batch_size = 4
n_epoch = 50
embedding_dim = 5
n_worker = 4

# load the data
x_vocab = Vocab(x_vocab_file, n_sense, id_offset=1)
y_vocab = Vocab(y_vocab_file, n_sense, id_offset=0)
train_data = RelationDataset(train_file, n_sense, feature_id_offset=1)
valid_data = RelationDataset(valid_file, n_sense, feature_id_offset=1)
test_data = RelationDataset(test_file, n_sense, feature_id_offset=1)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=n_worker)
valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=n_worker)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=n_worker)
x_vocab_size = x_vocab.size()
y_vocab_size = y_vocab.size()
train_data.gen_multinomial_dist(y_vocab_size / n_sense)

class SenseNS(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_vocab_size, n_sense):
        super(SenseNS, self).__init__()
        self.embeder_x = nn.Embedding(vocab_size, embedding_dim, padding_idx=constants.PAD)
        self.embeder_y = nn.Embedding(output_vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.n_sense = n_sense
    def forward(self, inputs, length_weights, word_attn_mask, y_inputs):
        mb_size, max_len = inputs.size()[0], inputs.size()[1] / self.n_sense
        y_len = y_inputs.size()[1] / self.n_sense
        embeds = self.embeder_x(inputs)
        y_embeds = self.embeder_y(y_inputs)
        # compute the context vector
        embeds = embeds.view(mb_size, max_len, self.n_sense, self.embedding_dim)
        embed_mean = torch.mean(embeds, dim=2).squeeze(2)  # mb_size * max_len * embedding_dim
        embed_mean = torch.sum(embed_mean, dim=1)  # mb_size * 1 * embedding_dim
        context_vec = torch.bmm(length_weights, embed_mean)  # mb_size * 1 * embedding_dim
        # compute the similarities for x embedding
        embeds = embeds.view(mb_size, -1, self.embedding_dim)  # mb_size * (max_len * n_sense) * embedding_dim
        context_vec = context_vec.transpose(1, 2)  # mb_size * embedding_dim * 1
        similarity_vec = torch.bmm(embeds, context_vec)  # mb_size * (max_len * n_sense) * 1
        # get the attention weights over the senses in x embedding
        attn_weights = similarity_vec.view(-1, self.n_sense)  # (mb_size * max_len) * n_sense
        attn_weights = F.softmax(attn_weights).view(mb_size, 1,  -1)  # mb_size * 1 * (max_len * n_sense)
        attn_weights = torch.bmm(length_weights, attn_weights).squeeze(1)  # scale by length, mb_size * (max_len * n_sense)
        attn_mask = self.get_attn_mask(inputs)
        attn_weights.data.masked_fill_(attn_mask, 0)  # mb_size * (max_len * n_sense)
        attn_weights = attn_weights.view(mb_size, 1, -1)  # mb_size * 1 * (max_len * n_sense)
        # now use the attention to get the hidden state: the weighted mean over x embeddings
        hidden = torch.bmm(attn_weights, embeds)  # mb_size * 1 * dim
        # compute the similarities between context vec and y embedding
        y_embeds = y_embeds.view(mb_size, y_len, self.n_sense, self.embedding_dim)  # mb_size * (y_len * n_sense) * embedding_dim
        y_word_mean = torch.mean(y_embeds, dim=2).squeeze(2).transpose(1, 2)
        # y_embeds = y_embeds.view(mb_size, -1, self.embedding_dim)  # mb_size * (y_len * n_sense) * embedding_dim
        # y_similarity_vec = torch.bmm(y_embeds, context_vec)  # mb_size * (y_len * n_sense) * 1
        # y_attn_weights = y_similarity_vec.view(-1, self.n_sense)  # (mb_size * y_len) * n_sense
        # y_attn_weights = F.softmax(y_attn_weights).view(mb_size * y_len, 1, self.n_sense)  # (mb_size * y_len) * 1 *  n_sense
        # y_word_mean = torch.bmm(y_attn_weights, y_embeds.view(mb_size * y_len, self.n_sense, self.embedding_dim)).squeeze(1) # (mb_size * y_len) * dim
        # y_word_mean = y_word_mean.view(mb_size, y_len, self.embedding_dim).transpose(1, 2)
        # compute final scores
        scores = torch.bmm(hidden, y_word_mean).squeeze(1)
        prob = F.sigmoid(scores)
        return prob
    def get_attn_mask(self, inputs):
        pad_attn_mask = inputs.data.eq(constants.PAD)   # mb_size x max_len
        return pad_attn_mask


def eval(model):
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

    num_correct, ranks, pool_ranks = 0, [], []
    for data_batch in test_loader:
        # get the input
        x_inputs = Variable(data_batch[0])
        length_weights = Variable(1.0/data_batch[1].float()).view(-1, 1, 1)
        word_masks = data_batch[2]
        ground_truth = data_batch[3]

        mb_size = x_inputs.size()[0]
        y_inputs = torch.linspace(0, y_vocab_size - 1, y_vocab_size).long()
        y_inputs = Variable(y_inputs.view(1, y_vocab_size).expand(mb_size, y_vocab_size))

        output = model(x_inputs, length_weights, word_masks, y_inputs)
        num_correct += get_num_correct(ground_truth, output.data)
        ranks.extend(get_groundtruth_rank_full(ground_truth, output.data))
        pool_ranks.extend(get_groundtruth_rank_pool(ground_truth, output.data))
    accuracy = float(num_correct) / float(len(test_data))
    mr_full = np.mean(ranks)
    mrr_full = np.mean([1.0/r for r in ranks])
    mr_pool = np.mean(pool_ranks)
    mrr_pool = np.mean([1.0/r for r in pool_ranks])
    return accuracy, mr_full, mrr_full, mr_pool, mrr_pool


def get_output(model, features, y_vocab_size):
    features = Variable(torch.LongTensor(features))
    y_labels = Variable(torch.LongTensor([i for i in xrange(y_vocab_size)]))
    return model(features, y_labels)


def main():
    criterion = NSNLLLoss()
    model = SenseNS(x_vocab_size, embedding_dim, y_vocab_size, n_sense)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.005, betas=(0.9, 0.98), eps=1e-09)
    # train
    best_accuracy = 0
    best_metrics = None
    total_time = 0
    for epoch in xrange(n_epoch):
        running_loss = 0.0
        start = time.time()
        for data_batch in train_loader:
            # get the input
            x_inputs = Variable(data_batch[0])
            length_weights = Variable(1.0/data_batch[1].float()).view(-1, 1, 1)
            word_masks = data_batch[2]
            labels = Variable(data_batch[3]).view(-1)
            y_inputs = Variable(data_batch[4])
            outputs = model(x_inputs, length_weights, word_masks, y_inputs)
            loss = criterion(outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]
        end = time.time()
        total_time += (end - start)
        print('[%d]  training loss: %.3f' % (epoch+1, running_loss))
        metrics = eval(model)
        accuracy = metrics[0]
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_metrics = metrics
        print metrics
    # metrics = evaluate_neg(test_data, model, y_vocab.size())
    print eval(model)
    print best_metrics
    print 'train time:', total_time

main()
