import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from loss import NSNLLLoss
import torch.optim as optim
from random import randint
import numpy.random as nprd

x_vocab_size = 10
y_vocab_size = 5
dim = 3

class Recon(nn.Module):
    def __init__(self):
        super(Recon, self).__init__()
        self.embedding = nn.Embedding(x_vocab_size, dim)
        self.linear = nn.Linear(dim, y_vocab_size)
    def forward(self, inputs):
        embeds = self.embedding(inputs)
        out = torch.mean(embeds, dim=0)
        out = self.linear(out)
        log_probs = F.log_softmax(out)
        return log_probs



class ReconBatch(nn.Module):
    def __init__(self):
        super(ReconBatch, self).__init__()
        self.embedding = nn.Embedding(x_vocab_size, dim, padding_idx=0)
        self.linear = nn.Linear(dim, y_vocab_size)
    def forward(self, inputs):
        embeds = self.embedding(inputs)
        out = torch.mean(embeds, dim=1).squeeze(1)
        out = self.linear(out)
        log_probs = F.log_softmax(out)
        return log_probs



class ReconNS(nn.Module):
    def __init__(self):
        super(ReconNS, self).__init__()
        self.embedding = nn.Embedding(x_vocab_size, dim)
        self.y_embedding = nn.Embedding(y_vocab_size, dim)
    def forward(self, inputs, labels):
        embeds = self.embedding(inputs)
        out = torch.mean(embeds, dim=0)
        y_embeds = self.y_embedding(labels)
        out = torch.mm(out, y_embeds.transpose(0,1))
        log_probs = F.sigmoid(out)
        # log_probs = F.log_softmax(out)
        return log_probs


# forward_time, backward_time = 0, 0
# model = Recon()
# criterion = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# for i in xrange(5000):
#     inputs = nprd.choice(x_vocab_size, 5)
#     inputs = Variable(torch.LongTensor(inputs))
#     label = randint(0, y_vocab_size - 1)
#     label = Variable(torch.LongTensor([label]))
#     f_start_time = time.time()
#     output = model(inputs)
#     f_end_time = time.time()
#     forward_time += f_end_time - f_start_time
#     loss = criterion(output, label)
#     optimizer.zero_grad()
#     b_start_time = time.time()
#     loss.backward()
#     optimizer.step()
#     b_end_time = time.time()
#     backward_time += (b_end_time - b_start_time)
# print forward_time, backward_time



forward_time, backward_time = 0, 0
model = ReconBatch()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
for i in xrange(1000):
    # inputs = [nprd.choice(x_vocab_size, 5) for i in xrange(2)]
    inputs = [[1,2,0], [2,3,5], [0,3,5], [0,4,5], [0,4,2]]
    inputs = Variable(torch.LongTensor(inputs))
    label = [randint(0, y_vocab_size - 1) for i in xrange(5)]
    label = Variable(torch.LongTensor(label))
    f_start_time = time.time()
    output = model(inputs)
    f_end_time = time.time()
    forward_time += f_end_time - f_start_time
    loss = criterion(output, label)
    optimizer.zero_grad()
    b_start_time = time.time()
    loss.backward()
    optimizer.step()
    b_end_time = time.time()
    backward_time += (b_end_time - b_start_time)
print forward_time, backward_time


# forward_time, backward_time = 0, 0
# model = ReconNS()
# criterion = NSNLLLoss()
# # criterion = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# ground_truth = Variable(torch.LongTensor([0]))
# for i in xrange(500):
#     inputs = nprd.choice(x_vocab_size, 5)
#     inputs = Variable(torch.LongTensor(inputs))
#     label = nprd.choice(y_vocab_size, 2)
#     label = Variable(torch.LongTensor(label))
#     f_start_time = time.time()
#     output = model(inputs, label)
#     f_end_time = time.time()
#     forward_time += f_end_time - f_start_time
#     loss = criterion(output)
#     # loss = criterion(output, ground_truth)
#     optimizer.zero_grad()
#     b_start_time = time.time()
#     loss.backward()
#     optimizer.step()
#     b_end_time = time.time()
#     backward_time += (b_end_time - b_start_time)
# print forward_time, backward_time
