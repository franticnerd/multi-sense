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
    pd = defaultdict()
    # for preprocessing
    pd['raw_data_file'] = '/Users/chao/data/source/tweets-dev/clean/tweets.txt'
    pd['grid_list'] = [100, 100]
    pd['train_ratio'] = 0.8
    pd['valid_ratio'] = 0.1
    pd['min_token_freq'] = 5
    pd['classify_train_ratio'] = 0.8
    # for model training and evaluation
    pd['data_dir'] = '../data/toy/'
    pd['load_pretrained'] = False
    pd['load_model'] = False
    pd['save_model'] = True
    pd['regu_strength'] = 1e-4
    pd['dropout'] = 0
    pd['data_worker'] = 1
    pd['n_sense'] = 2
    pd['embedding_dim'] = 2
    pd['batch_size'] = 4
    pd['n_epoch'] = 5
    pd['print_gap'] = 20
    pd['learning_rate'] = 0.005
    pd['eval_lr'] = False
    pd['eval_batch'] = False
    pd['eval_dim'] = False
    pd['eval_sense'] = False
    # pd['model_type_list'] = ['recon', 'attn', 'sense', 'attn_sense', 'bilinear_sense', 'comp_attn_sense']
    # pd['model_type_list'] = ['attn_sense', 'recon', 'attn', 'sense', 'bilinear_sense', 'comp_attn_sense']
    pd['model_type_list'] = ['recon', 'sense']
    pd['cmp_model_type_list'] = ['recon', 'sense']
    return pd

def print_config(pd):
    content = [('model_type:', pd['model_type']),
               ('data_dir:', pd['data_dir']),
               ('embedding_dim:', pd['embedding_dim']),
               ('n_epoch:', pd['n_epoch'])]
    print format_list_to_string(content)

