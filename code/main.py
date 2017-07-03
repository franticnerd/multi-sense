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

torch.manual_seed(1)


def load_data(train_data_file, test_data_file, x_vocab_file, y_vocab_file):
    train_data = RelationDataset(train_data_file)
    test_data = RelationDataset(test_data_file)
    x_vocab = Vocab(x_vocab_file)
    y_vocab = Vocab(y_vocab_file)
    return train_data, test_data, x_vocab, y_vocab


def build_cbow_model(x_vocab_size, embedding_dim, y_vocab_size):
    return CBOW(x_vocab_size, embedding_dim, y_vocab_size)


def main(pd):
    train_data, test_data, x_vocab, y_vocab = load_data(pd['train_data_file'],
                                                        pd['test_data_file'],
                                                        pd['x_vocab_file'],
                                                        pd['y_vocab_file'])
    cbow = build_cbow_model(x_vocab.size(), pd['embedding_dim'], y_vocab.size())
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(cbow.parameters(), lr=0.001, momentum=0.9)
    n_epoch = pd['n_epoch']

    start = time.time()
    # train
    for epoch in xrange(n_epoch):
        running_loss = 0.0
        for i in xrange(len(train_data)):
            # get the input
            inputs, labels = train_data[i]
            inputs = Variable(torch.LongTensor(inputs))
            labels = Variable(torch.LongTensor(labels))
            # forward
            outputs = cbow(inputs)
            loss = criterion(outputs, labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            # backward + optimize
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]
            if (i + 1) % 2000 == 0:
                print('[%d, %5d]  training loss: %.3f' % (epoch+1, i+1, running_loss/2000))
                running_loss = 0.0
    end = time.time()
    print 'Total training time:', end - start
    # evaluate
    metrics = evaluate(test_data, cbow)
    print 'Performance on test data: ', '\t'.join(str(e) for e in metrics)


if __name__ == '__main__':
    para_file = None if len(sys.argv) <= 1 else sys.argv[1]
    pd = load_params(para_file)  # load parameters as a dict
    main(pd)
