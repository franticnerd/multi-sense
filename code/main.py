import sys

from dataset import DataSet
from evaluator import Evaluator
from analyzer import CaseAnalyzer
from model_manager import ModelManager
from paras import load_params
from utils import set_random_seeds
from instance_analyzer import InstanceAnalyzer

def run_one_config(opt, model_type, case_study=False):
    set_random_seeds()
    dataset = DataSet(opt, model_type)
    model_manager = ModelManager(opt)
    model, train_time = model_manager.build_model(model_type, dataset)
    evaluator = Evaluator(opt)
    metrics = evaluator.eval(model, model_type, dataset.test_loader)
    evaluator.write_performance(model_type, metrics, train_time)
    run_case_study(model, dataset, opt, case_study)

def run_case_study(model, dataset, opt, case_study):
    if not case_study:
        return
    case_evaluator = CaseAnalyzer(model, dataset, opt)
    case_evaluator.find_duplicate_sense_words()
    case_evaluator.find_similar_words()

def run_error_analysis(opt):
    instance_analyzer = InstanceAnalyzer(opt)
    instance_analyzer.write_error_results(opt)

def eval_batch_size(opt, model_type):
    if not opt['eval_batch']:
        return
    default_batch_size = opt['batch_size']
    for batch_size in opt['batch_list']:
        opt['batch_size'] = batch_size
        run_one_config(opt, model_type)
    opt['batch_size'] = default_batch_size


def eval_learning_rate(opt, model_type):
    if not opt['eval_lr']:
        return
    default_lr = opt['learning_rate']
    for lr in opt['lr_list']:
        opt['learning_rate'] = lr
        run_one_config(opt, model_type)
    opt['learning_rate'] = default_lr


def eval_embedding_dim(opt, model_type):
    if not opt['eval_dim']:
        return
    default_dim = opt['embedding_dim']
    for dim in opt['dim_list']:
        opt['embedding_dim'] = dim
        run_one_config(opt, model_type)
    opt['embedding_dim'] = default_dim

def eval_n_sense(opt, model_type):
    if not opt['eval_sense']:
        return
    default_n_sense = opt['n_sense']
    for n_sense in opt['n_sense_list']:
        opt['n_sense'] = n_sense
        run_one_config(opt, model_type)
    opt['n_sense'] = default_n_sense


def eval_dropout(opt, model_type):
    if not opt['eval_dp']:
        return
    default_dp = opt['dropout']
    for dropout in opt['dp_list']:
        opt['dropout'] = dropout
        run_one_config(opt, model_type)
    opt['dropout'] = default_dp

def eval_regularization(opt, model_type):
    if not opt['eval_regu']:
        return
    default_regu = opt['regu_strength']
    for regu in opt['regu_list']:
        opt['regu_strength'] = regu
        run_one_config(opt, model_type)
    opt['regu_strength'] = default_regu

def main(opt):
    for model_type in opt['model_type_list']:
        run_one_config(opt, model_type, True)
        # eval_learning_rate(opt, model_type)
        # eval_batch_size(opt, model_type)
        # eval_embedding_dim(opt, model_type)
        # eval_n_sense(opt, model_type)
        eval_dropout(opt, model_type)
        eval_regularization(opt, model_type)
    run_error_analysis(opt)

if __name__ == '__main__':
    # para_file = '../scripts/la-10k.yaml'
    para_file = None if len(sys.argv) <= 1 else sys.argv[1]
    opt = load_params(para_file)  # load parameters as a dict
    main(opt)
