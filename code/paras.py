import yaml
from collections import defaultdict
from utils import format_list_to_string

class yaml_loader:
    def __init__(self):
        pass

    def load(self, para_file):
        yaml.add_constructor('!join', self._concat)
        fin = open(para_file, 'r')
        # using default dict: if the key is not specified, the values is None
        return defaultdict(lambda: None, yaml.load(fin))

    def _concat(self, loader, node):
        seq = loader.construct_sequence(node)
        return ''.join([str(i) for i in seq])


def load_params(para_file):
    if para_file is None:
        para = set_default_params()
    else:
        para = yaml_loader().load(para_file)
    return para


def set_default_params():
    pd = dict()
    pd['data_dir'] = '../data/toy/'
    # for preprocessing
    pd['raw_data_file'] = '/Users/chao/data/source/tweets-dev/clean/tweets.txt'
    pd['grid_list'] = [100, 100]
    pd['train_ratio'] = 0.8
    pd['valid_ratio'] = 0.1
    pd['min_token_freq'] = 5
    # for training
    pd['train_data_file'] = pd['data_dir'] + 'input/train.txt'
    pd['test_data_file'] = pd['data_dir'] + 'input/test.txt'
    pd['valid_data_file'] = pd['data_dir'] + 'input/test.txt'
    pd['x_vocab_file'] = pd['data_dir'] + 'input/words.txt'
    pd['y_vocab_file'] = pd['data_dir'] + 'input/locations.txt'
    pd['train_log_file'] = pd['data_dir'] + 'output/train_log.txt'
    pd['performance_file'] = pd['data_dir'] + 'output/performance.txt'
    pd['case_seed_file'] = pd['data_dir'] + 'input/case_seeds.txt'
    pd['case_output_file'] = pd['data_dir'] + 'output/case_outputs.txt'
    pd['model_path'] = pd['data_dir'] + 'model/'
    pd['n_sense'] = 2
    pd['K'] = 2
    # model and training opts
    # pd['model_type_list'] = ['comp_attn_sense_net']
    # pd['model_type_list'] = ['cbow', 'comp_attn_sense_net']
    pd['model_type_list'] = ['sense']
    # pd['model_type_list'] = ['bi_sense_net']
    # pd['model_type'] = 'attn_net'
    pd['embedding_dim'] = 2
    pd['n_epoch'] = 10
    pd['print_gap'] = 20
    return pd


def print_config(pd):
    content = [('model_type:', pd['model_type']),
               ('data_dir:', pd['data_dir']),
               ('embedding_dim:', pd['embedding_dim']),
               ('n_epoch:', pd['n_epoch'])]
    print format_list_to_string(content)

