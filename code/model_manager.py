import torch

from utils import format_list_to_string, ensure_directory_exist
from models.model import Recon, AttnNet, SenseNet, AttnSenseNet, CompAttnSenseNet, BilinearSenseNet, BidirectionSenseNet


class ModelManager:
    def __init__(self, opt):
        self.opt = opt

    def load_model(self, model, model_type):
        model_name = self.get_model_name(model_type)
        file_name = self.opt['model_path'] + model_name
        model.load_state_dict(torch.load(file_name))

    def save_model(self, model, model_type):
        model_name = self.get_model_name(model_type)
        file_name = self.opt['model_path'] + model_name
        ensure_directory_exist(file_name)
        torch.save(model.state_dict(), file_name)

    def get_model_name(self, model_type):
        embedding_dim = self.opt['embedding_dim']
        attributes = [model_type, embedding_dim]
        model_name = format_list_to_string(attributes, '_')
        return model_name + '.model'

    def build_model(self, model_type, x_vocab_size, y_vocab_size):
        embedding_dim = self.opt['embedding_dim']
        n_sense = self.opt['n_sense']
        if model_type == 'recon':
            return Recon(x_vocab_size, embedding_dim, y_vocab_size)
        elif model_type == 'attn':
            return AttnNet(x_vocab_size, embedding_dim, y_vocab_size)
        elif model_type == 'sense':
            return SenseNet(x_vocab_size, embedding_dim, y_vocab_size, n_sense)
        elif model_type == 'bilinear_sense':
            return BilinearSenseNet(x_vocab_size, embedding_dim, y_vocab_size, n_sense)
        elif model_type == 'bidirection_sense':
            return BidirectionSenseNet(x_vocab_size, embedding_dim, y_vocab_size, n_sense)
        elif model_type == 'attn_sense':
            return AttnSenseNet(x_vocab_size, embedding_dim, y_vocab_size, n_sense)
        elif model_type == 'comp_attn_sense':
            return CompAttnSenseNet(x_vocab_size, embedding_dim, y_vocab_size, n_sense)
        else:
            print 'Model type not supported!'
            return None
