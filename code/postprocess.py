import numpy as np
from collections import Counter

def count(instance_file):
    len_list = []
    with open(instance_file, 'r') as fin:
        for line in fin:
            words = line.strip().split()
            len_list.append(len(words))
    return len_list

# f = '/Users/chao/data/projects/multi-sense-embedding/tweets-1m/output/instance-recon-0-comp_attn_sense-1-.txt'
f = '/Users/chao/data/projects/multi-sense-embedding/checkins-ny/output/instance-recon-0-comp_attn_sense-0-.txt'
lens = count(f)
print np.mean(lens)
c = Counter(lens)
max_len = np.max(c.keys())
for i in xrange(1, max_len + 1):
    print i, c[i]
