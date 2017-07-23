import sys

from dataset import DataSet
from evaluator import Evaluator, CaseEvaluator
from model_manager import ModelManager
from paras import load_params
from utils import set_random_seeds


def run_one_config(opt, model_type, case_study=False):
    set_random_seeds()
    dataset = DataSet(opt, model_type)
    model_manager = ModelManager(opt)
    model, train_time = model_manager.build_model(model_type, dataset)
    evaluator = Evaluator(opt)
    metrics = evaluator.eval(model, dataset.test_loader)
    evaluator.write_performance(model_type, metrics, train_time)
    if case_study:
        case_evaluator = CaseEvaluator(model, dataset, opt)
        case_evaluator.run_case_study()
    # evaluator.eval_classification(opt, model_type, model)

def eval_error_analysis(opt):
    evaluator = Evaluator(opt)
    evaluator.write_error_results(opt)

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

def main(opt):
    for model_type in opt['model_type_list']:
        run_one_config(opt, model_type, True)
        # run_one_config(opt, model_type, True)
        eval_learning_rate(opt, model_type)
        eval_batch_size(opt, model_type)
        eval_embedding_dim(opt, model_type)
        # eval_n_sense(opt, model_type)
    eval_error_analysis(opt)

if __name__ == '__main__':
    para_file = None if len(sys.argv) <= 1 else sys.argv[1]
    opt = load_params(para_file)  # load parameters as a dict
    main(opt)
