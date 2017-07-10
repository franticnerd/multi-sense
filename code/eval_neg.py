import numpy as np
import torch
from torch.autograd import Variable
import random
from random import randint
from utils import format_list_to_string


def evaluate_neg(test_data, model, y_vocab_size):
    random.seed(1)
    num_correct, ranks, pool_ranks = 0, [], []
    for i in xrange(len(test_data)):
        features, ground_truth = test_data[i]
        ground_truth = torch.LongTensor(ground_truth)
        output = get_output(model, features, y_vocab_size)

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
    ret = []
    for i in xrange(y_vocab_size):
        test_label = Variable(torch.LongTensor([i]))
        score = model(Variable(torch.LongTensor(features)), test_label)
        ret.append(score.data[0])
    return Variable(torch.Tensor(ret)).view(1, -1)



# check whether the ground truth appears in the top-K list, for computing the hit ratio
def get_num_correct(ground_truth, output):
    correct = 0
    _, predicted = torch.max(output, 1)
    correct += (predicted == ground_truth).sum()
    return correct


def get_groundtruth_rank_full(ground_truth, output):
    ret = []
    for true_id, predicted in zip(ground_truth, output):
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


def print_metrics(accuracy, mr_full, mrr_full, mr_pool, mrr_pool):
    content = [('accuracy:', accuracy),
               ('mr_full:', mr_full),
               ('mrr_full:', mrr_full),
               ('mr_pool:', mr_pool),
               ('mrr_pool:', mrr_pool)]
    print format_list_to_string(content)
