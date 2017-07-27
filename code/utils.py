import numpy as np
import os
import random
import torch

# convert a list into a string
def format_list_to_string(l, sep='\n'):
    ret = []
    for e in l:
        if type(e) == float or type(e) == np.float64:
            ret.append(format_float_to_string(e))
        elif type(e) == list or type(e) == tuple:
            # ret.append(format_list_to_string(e, '\t'))
            ret.append(format_list_to_string(e, ' '))
        else:
            ret.append(str(e))
    return sep.join(ret)


def format_float_to_string(f):
    return str.format('{0:.4f}', f)


# ensure the path for the output file exist
def ensure_directory_exist(file_name):
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)


def read_first_line(perf_file):
    if not os.path.exists(perf_file):
        return None
    with open(perf_file, 'r') as fin:
        first_line = fin.readline().strip()
        return first_line


def calc_sim(vec_a, vec_b, mode='dot'):
    if mode == 'dot':
        return torch.dot(vec_a, vec_b)
    elif mode == 'cosine':
        return calc_cosine(vec_a, vec_b)


def calc_cosine(vec_a, vec_b):
    norm_prod = vec_a.norm() * vec_b.norm()
    denominator = np.max([1e-8, norm_prod])
    return np.dot(vec_a.tolist(), vec_b.tolist()) / denominator


# given a list of scores, find the top k similar ones and the idx
def find_topk_neighbors(similarities, K):
    scores, neighbor_idx = torch.topk(similarities, K)
    return scores.tolist(), neighbor_idx.tolist()

def append_to_file(output_file, content):
    ensure_directory_exist(output_file)
    with open(output_file, 'a') as fp:
        fp.write(content + '\n')


def set_random_seeds():
    torch.manual_seed(1)
    random.seed(1)


# x = torch.FloatTensor([0,1])
# y = torch.FloatTensor([0,-1])
# print x
# print y
# print calc_cosine(x, y)
