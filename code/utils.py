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
            ret.append(format_list_to_string(e, '\t'))
        else:
            ret.append(str(e))
    return sep.join(ret)


def format_float_to_string(f):
    return str.format('{0:.3f}', f)


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


def set_random_seeds():
    torch.manual_seed(1)
    random.seed(1)
