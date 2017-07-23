import sys
from zutils.dto.text.word_dict import WordDict
from paras import load_params
from collections import Counter
from random import shuffle
from utils import ensure_directory_exist
from utils import format_list_to_string

def load_text(input_file):
    with open(input_file, 'r') as fin:
        content = fin.read()
    return content.strip().split()

def trim_word_set(word_counter, min_count=5):
    word_set = set()
    for w, c in word_counter.items():
        if c >= min_count:
            word_set.add(w)
    return word_set

def build_intances(words, word_set, window_size=5):
    instances = [(words[i:i+window_size], words[i+window_size]) for i in xrange(len(words)-window_size)]
    trimmed_instances = []
    for instance in instances:
        feature = [word for word in instance[0] if word in word_set]
        label = instance[1] if instance[1] in word_set else None
        if len(feature) > 1 and label is not None:
            trimmed_instances.append((feature, label))
    return trimmed_instances


def build_vocab(instances):
    x_vocab = WordDict()
    y_vocab = WordDict()
    for instance in instances:
        features = instance[0]
        label = instance[1]
        x_vocab.update_count(features)
        y_vocab.update_count([label])
    x_vocab.encode_words(min_freq=0)
    y_vocab.encode_words(min_freq=0)
    return x_vocab, y_vocab


def split_data(instances, train_ratio=0.8):
    shuffle(instances)
    n_train = int(len(instances) * train_ratio)
    return instances[:n_train], instances[n_train:]


def write_to_file(instances, x_vocab, y_vocab, output_file):
    ensure_directory_exist(output_file)
    with open(output_file, 'w') as fout:
        for instance in instances:
            instance_string = format_instance_to_string(instance, x_vocab, y_vocab)
            fout.write(instance_string + '\n')

def format_instance_to_string(instance, x_vocab, y_vocab):
    content = []
    features = [x_vocab.get_word_id(e) for e in instance[0]]
    label = y_vocab.get_word_id(instance[1])
    content.append(label)
    content.append(features)
    return format_list_to_string(content, '\t')

def run(opt):
    words = load_text(opt['raw_data_file'])
    word_counter = Counter(words)
    word_set = trim_word_set(word_counter, opt['min_token_freq'])
    instances = build_intances(words, word_set)
    x_vocab, y_vocab = build_vocab(instances)
    train_data, test_data = split_data(instances)
    write_to_file(train_data, x_vocab, y_vocab, opt['train_data_file'])
    write_to_file(test_data, x_vocab, y_vocab, opt['test_data_file'])
    x_vocab.write_to_file(opt['x_vocab_file'])
    y_vocab.write_to_file(opt['y_vocab_file'])

if __name__ == '__main__':
    para_file = None if len(sys.argv) <= 1 else sys.argv[1]
    pd = load_params(para_file) # load parameters as a dict
    run(pd)
