import pandas as pd

from dataset import Vocab
from utils import format_list_to_string


class InstanceAnalyzer:

    def __init__(self, opt):
        self.test_data_file = opt['data_dir'] + 'input/test.txt'
        self.x_vocab_file = opt['data_dir'] + 'input/words.txt'
        self.instance_analysis_path = opt['data_dir'] + 'instance/'

    def write_error_results(self, opt):
        model_a_name, model_b_name = opt['cmp_model_type_list']
        model_a_file = self.instance_analysis_path + model_a_name + '.txt'
        model_b_file = self.instance_analysis_path + model_b_name + '.txt'
        model_a_indicator = pd.read_table(model_a_file, header=None).ix[:, 0]
        model_b_indicator = pd.read_table(model_b_file, header=None).ix[:, 0]
        test_instances = pd.read_table(self.test_data_file, header=None).as_matrix()
        vocab = Vocab(self.x_vocab_file, n_sense=1, id_offset=0)
        instances = self.select_better_instances(test_instances, model_a_indicator, model_b_indicator)
        # where model b makes correct predictions but model a does not
        output_file = self.instance_analysis_path + model_a_name + '-0-' + model_b_name + '-1-.txt'
        self.write_model_instances(instances, vocab, output_file)
        instances = self.select_better_instances(test_instances, model_b_indicator, model_a_indicator)
        # where model a makes correct predictions but model b does not
        output_file = self.instance_analysis_path + model_a_name + '-1-' + model_b_name + '-0-.txt'
        self.write_model_instances(instances, vocab, output_file)
        # where both model a and b make correct predictions
        instances = self.select_equal_instances(test_instances, model_a_indicator, model_b_indicator, 1)
        output_file = self.instance_analysis_path + model_a_name + '-1-' + model_b_name + '-1-.txt'
        self.write_model_instances(instances, vocab, output_file)
        # where both model a and b make wrong predictions
        instances = self.select_equal_instances(test_instances, model_a_indicator, model_b_indicator, 0)
        output_file = self.instance_analysis_path + model_a_name + '-0-' + model_b_name + '-0-.txt'
        self.write_model_instances(instances, vocab, output_file)

        # select the instances where b is better than a
    def select_better_instances(self, instances, indicator_a, indicator_b):
        b_better_instances = []
        for i in xrange(instances.shape[0]):
            if indicator_b[i] > indicator_a[i]:
                b_better_instances.append(instances[i])
        return b_better_instances

    def select_equal_instances(self, instances, indicator_a, indicator_b, target):
        ret = []
        for i in xrange(instances.shape[0]):
            if indicator_a[i] == target and indicator_b[i] == target:
                ret.append(instances[i])
        return ret

    def write_model_instances(self, instances, vocab, output_file):
        with open(output_file, 'w') as fout:
            for e in instances:
                instance_string = self.get_instance_string(e, vocab)
                fout.write(instance_string + '\n')

    def get_instance_string(self, instance, vocab):
        feature_idx = instance[1].split()
        feature_idx = [int(e) for e in feature_idx]
        instance_attributes = [vocab.get_description(idx) for idx in feature_idx]
        return format_list_to_string(instance_attributes, sep=' ')
