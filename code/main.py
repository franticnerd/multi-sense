import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from paras import load_params
from evaluator import evaluate
from dataset import RelationDataset, Vocab
from models.cbow import CBOW
from models.attn_net import AttnNet, SenseNet

torch.manual_seed(1)


def load_data(pd):
    train_data = RelationDataset(pd['train_data_file'], pd['multi_sense'], pd['n_sense'])
    test_data = RelationDataset(pd['test_data_file'], pd['multi_sense'], pd['n_sense'])
    x_vocab = Vocab(pd['x_vocab_file'], pd['multi_sense'], pd['n_sense'])
    y_vocab = Vocab(pd['y_vocab_file'], pd['multi_sense'], pd['n_sense'])
    return train_data, test_data, x_vocab, y_vocab


def build_model(x_vocab_size, y_vocab_size, pd):
    embedding_dim = pd['embedding_dim']
    if pd['model_type'] == 'cbow':
        return CBOW(x_vocab_size, embedding_dim, y_vocab_size)
    elif pd['model_type'] == 'attn_net':
        return AttnNet(x_vocab_size, embedding_dim, y_vocab_size)
    elif pd['model_type'] == 'sense_net':
        return SenseNet(x_vocab_size, embedding_dim, y_vocab_size, pd['n_sense'])
    else:
        print 'Model type not supported!'
        return None


def train(train_data, model, criterion, optimizer, pd):
    n_epoch = pd['n_epoch']
    # train
    for epoch in xrange(n_epoch):
        running_loss = 0.0
        for i in xrange(len(train_data)):
            # get the input
            inputs, labels = train_data[i]
            inputs = Variable(torch.LongTensor(inputs))
            labels = Variable(torch.LongTensor(labels))
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # backward + optimize
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]
            if (i + 1) % 500 == 0:
                print('[%d, %5d]  training loss: %.3f' % (epoch+1, i+1, running_loss/500))
                running_loss = 0.0


def print_parameters(model):
    for p in model.parameters():
        print p


def main(pd):
    train_data, test_data, x_vocab, y_vocab = load_data(pd)
    model = build_model(x_vocab.size(), y_vocab.size(), pd)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    start = time.time()
    train(train_data, model, criterion, optimizer, pd)
    end = time.time()
    print 'Total training time:', end - start

    # evaluate
    metrics = evaluate(test_data, model)
    print 'Performance on test data: ', '\t'.join(str(e) for e in metrics)


if __name__ == '__main__':
    para_file = None if len(sys.argv) <= 1 else sys.argv[1]
    pd = load_params(para_file)  # load parameters as a dict
    main(pd)