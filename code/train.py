import time
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import ensure_directory_exist
from evaluator import Evaluator


class Trainer:

    def __init__(self, model, opt, model_type):
        self.opt = opt
        self.train_log_file = opt['train_log_file']
        ensure_directory_exist(self.train_log_file)
        self.n_epoch = opt['n_epoch']
        self.batch_size = opt['batch_size']  # mini batch size
        self.print_gap = opt['print_gap']
        self.save_model = opt['save_model']
        learning_rate = opt['learning_rate']
        self.evaluator = Evaluator(opt)
        self.model_type = model_type
        # init the loss and optimizer
        self.model = model
        self.criterion = nn.NLLLoss()
        # self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.01, betas=(0.9, 0.98), eps=1e-09)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.85, 0.98), eps=1e-09)

    def train(self, train_data, validation_data, model_manager):
        best_accuracy = 0
        start = time.time()
        for epoch in xrange(self.n_epoch):
            self.train_one_epoch(train_data, epoch)
            if self.save_model:
                valid_accuracy = self.validate_one_epoch(validation_data, epoch)
                if valid_accuracy >= best_accuracy:
                    best_accuracy = valid_accuracy
                    model_manager.save_model(self.model, self.model_type)
        end = time.time()
        return end - start

    def train_one_epoch(self, train_data, epoch):
        running_loss = 0.0
        for i, data_batch in enumerate(train_data):
            # get the input
            inputs = Variable(data_batch[0])
            length_weights = Variable(1.0/data_batch[1].float()).view(-1, 1, 1)
            word_masks = data_batch[2]
            labels = Variable(data_batch[3]).view(-1)
            # forward
            outputs = self.model(inputs, length_weights, word_masks)
            loss = self.criterion(outputs, labels)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            # backward + optimize
            loss.backward()
            self.optimizer.step()
            # print statistics
            running_loss += loss.data[0]
            if (i + 1) % self.print_gap == 0:
                self.write_train_loss(epoch, i, running_loss)
                running_loss = 0.0

    def write_train_loss(self, epoch, n_batch, running_loss):
        loss_info = '%20s [%d, %5d]  training loss: %.3f' % \
                    (self.model_type, epoch + 1, n_batch + 1, running_loss/self.print_gap)
        print loss_info
        with open(self.train_log_file, 'a') as fp:
            fp.write(loss_info + '\n')

    # validate the model after each epoch, return the accuracy
    def validate_one_epoch(self, validation_data, epoch):
        # metrics = self.evaluator.eval(self.model, validation_data)
        accuracy = self.evaluator.eval_accuracy(self.model, validation_data)
        validation_info = '%20s [%d]  validation accuracy: %.3f' % \
                          (self.model_type, epoch + 1, accuracy)
        with open(self.train_log_file, 'a') as fp:
            fp.write(validation_info + '\n')
        return accuracy

