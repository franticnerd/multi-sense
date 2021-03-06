import os
import sys
import time

import numpy.random as nprd
import torch
import torch.optim as optim
from torch.autograd import Variable

from dataset import RelationData, Vocab
from eval_neg import evaluate_neg
from loss import NSNLLLoss
from models.model import AttnNet, SenseNet, AttnSenseNet, CompAttnSenseNet
from models.model import ReconNS
from paras import load_params
from utils import format_list_to_string, ensure_directory_exist


def load_data(model_type, pd):
    multi_sense, n_sense = set_sense_paras(model_type, pd)
    x_vocab = Vocab(pd['x_vocab_file'], multi_sense, n_sense)
    y_vocab = Vocab(pd['y_vocab_file'], False, 1)
    train_data = RelationData(pd['train_data_file'], multi_sense, n_sense)
    test_data = RelationData(pd['test_data_file'], multi_sense, n_sense)
    train_data.gen_multinomial_dist(y_vocab.size())
    return train_data, test_data, x_vocab, y_vocab


# set the sense parameters based on model type
def set_sense_paras(model_type, pd):
    if model_type in ['cbow', 'attn_net']:
        return False, 1
    elif model_type in ['sense_net', 'attn_sense_net', 'comp_attn_sense_net']:
        return True, pd['n_sense']


def build_model(x_vocab_size, y_vocab_size, model_type, pd):
    embedding_dim = pd['embedding_dim']
    if model_type == 'cbow':
        return ReconNS(x_vocab_size, embedding_dim, y_vocab_size)
    elif model_type == 'attn_net':
        return AttnNet(x_vocab_size, embedding_dim, y_vocab_size)
    elif model_type == 'sense_net':
        return SenseNet(x_vocab_size, embedding_dim, y_vocab_size, pd['n_sense'])
    elif model_type == 'attn_sense_net':
        return AttnSenseNet(x_vocab_size, embedding_dim, y_vocab_size, pd['n_sense'])
    elif model_type == 'comp_attn_sense_net':
        return CompAttnSenseNet(x_vocab_size, embedding_dim, y_vocab_size, pd['n_sense'])
    else:
        print 'Model type not supported!'
        return None




def train_neg(train_data, model, criterion, optimizer, model_type, pd):
    forward_time, backward_time = 0, 0
    n_epoch = pd['n_epoch']
    train_log_file = pd['train_log_file']
    ensure_directory_exist(train_log_file)
    with open(train_log_file, 'a') as fout:
        # train
        for epoch in xrange(n_epoch):
            running_loss = 0.0
            for i in xrange(len(train_data)):
                # get the input
                inputs, labels = train_data[i]
                inputs = Variable(torch.LongTensor(inputs))
                noise_labels = train_data.sample_negatives(5, labels[0])
                labels.extend(noise_labels)
                labels = Variable(torch.LongTensor(labels))

                f_start_time = time.time()
                output = model(inputs, labels)
                f_end_time = time.time()
                forward_time += f_end_time - f_start_time

                loss = criterion(output)
                # zero the parameter gradients
                optimizer.zero_grad()
                # backward + optimize

                b_start_time = time.time()
                loss.backward()
                optimizer.step()
                b_end_time = time.time()
                backward_time += (b_end_time - b_start_time)

                # print statistics
                running_loss += loss.data[0]
                if (i + 1) % 2000 == 0:
                    print('%20s [%d, %5d]  training loss: %.3f' % (model_type, epoch+1, i+1, running_loss/2000))
                    fout.write('%20s [%d, %5d]  training loss: %.3f\n' % (model_type, epoch+1, i+1, running_loss/20))
                    running_loss = 0.0
    print 'forward time:', forward_time
    print 'backward time:', backward_time


def write_performance(pd, model_type, metrics, train_time):
    header = ['time', 'acc', 'mr_f', 'mrr_f', 'mr_p', 'mrr_p',\
              'n_sense', 'dim', 'n_epoch', 'data_dir', 'model_type']
    content = [train_time, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4],\
               pd['n_sense'], pd['embedding_dim'], pd['n_epoch'], pd['data_dir'], model_type]
    header_string = format_list_to_string(header, '\t')
    content_string = format_list_to_string(content, '\t')
    print header_string + '\n' + content_string
    # write to file
    perf_file = pd['performance_file']
    ensure_directory_exist(perf_file)
    file_header = read_first_line(perf_file)
    with open(perf_file, 'a') as fout:
        if file_header != header_string:
            fout.write(header_string + '\n')
        fout.write(content_string + '\n')


def read_first_line(perf_file):
    if not os.path.exists(perf_file):
        return None
    with open(perf_file, 'r') as fin:
        first_line = fin.readline().strip()
        return first_line


# negative sampling
def main(pd):
    for model_type in pd['model_type_list']:
        torch.manual_seed(1)
        nprd.seed(1)
        train_data, test_data, x_vocab, y_vocab = load_data(model_type, pd)
        model = build_model(x_vocab.size(), y_vocab.size(), model_type, pd)
        # criterion = nn.BCELoss()
        # criterion = nn.NLLLoss()
        criterion = NSNLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # train
        start = time.time()
        train_neg(train_data, model, criterion, optimizer, model_type, pd)
        end = time.time()
        train_time = end - start
        # # evaluate
        metrics = evaluate_neg(test_data, model, y_vocab.size())
        write_performance(pd, model_type, metrics, train_time)



if __name__ == '__main__':
    para_file = None if len(sys.argv) <= 1 else sys.argv[1]
    pd = load_params(para_file)  # load parameters as a dict
    main(pd)




# def train(train_data, model, criterion, optimizer, model_type, pd):
#     n_epoch = pd['n_epoch']
#     train_log_file = pd['train_log_file']
#     ensure_directory_exist(train_log_file)
#     with open(train_log_file, 'a') as fout:
#         # train
#         for epoch in xrange(n_epoch):
#             running_loss = 0.0
#             for i in xrange(len(train_data)):
#                 # get the input
#                 inputs, label = train_data[i]
#                 inputs = Variable(torch.LongTensor(inputs))
#                 noise_labels = train_data.sample_negatives(5, label)
#                 output = model(inputs, Variable(torch.LongTensor(label)))
#                 target = torch.autograd.Variable(torch.Tensor([0.99]))
#                 loss = criterion(output, target)
#                 # zero the parameter gradients
#                 optimizer.zero_grad()
#                 # backward + optimize
#                 loss.backward()
#                 optimizer.step()
#                 # print statistics
#                 running_loss += loss.data[0]
#                 xx = output.data[0]
#                 for nl in noise_labels:
#                     output = model(inputs, Variable(torch.LongTensor([nl])))
#                     yy = output.data[0]
#                     loss = criterion(output, torch.autograd.Variable(torch.Tensor([0.01])))
#                     # zero the parameter gradients
#                     optimizer.zero_grad()
#                     # backward + optimize
#                     loss.backward()
#                     optimizer.step()
#                     # print statistics
#                     running_loss += loss.data[0]
#                 if (i + 1) % 2000 == 0:
#                     print xx, yy
#                     print('%20s [%d, %5d]  training loss: %.3f' % (model_type, epoch+1, i+1, running_loss/2000))
#                     fout.write('%20s [%d, %5d]  training loss: %.3f\n' % (model_type, epoch+1, i+1, running_loss/20))
#                     running_loss = 0.0
