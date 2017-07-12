import sys

from dataset import DataSet
from evaluator import Evaluator, CaseEvaluator
from model_manager import ModelManager
from paras import load_params
from train import Trainer
from utils import set_random_seeds


def main(opt):
    for model_type in opt['model_type_list']:
        set_random_seeds()
        dataset = DataSet(opt, model_type)
        model_manager = ModelManager(opt)
        model = model_manager.build_model(model_type, dataset.x_vocab.size(), dataset.y_vocab.size())
        trainer = Trainer(model, opt, model_type)
        train_time = trainer.train(dataset.train_data, dataset.valid_data, model_manager)
        evaluator = Evaluator(opt)
        model_manager.load_model(model, model_type)  # load the best model for evaluation
        metrics = evaluator.eval(model, dataset.test_data)
        evaluator.write_performance(model_type, metrics, train_time)
        case_evaluator = CaseEvaluator(model, dataset, opt)
        case_evaluator.run_case_study()


if __name__ == '__main__':
    para_file = None if len(sys.argv) <= 1 else sys.argv[1]
    opt = load_params(para_file)  # load parameters as a dict
    main(opt)
