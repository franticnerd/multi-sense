import time
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import append_to_file
from evaluator import Evaluator
from utils import format_list_to_string


class Trainer:

    def __init__(self, model, opt, model_type):
        self.opt = opt
        self.train_log_file = opt['data_dir'] + 'output/train.log'
        self.valid_log_file = opt['data_dir'] + 'output/train.log'
        self.n_epoch = opt['n_epoch']
        self.batch_size = opt['batch_size']  # mini batch size
        self.print_gap = opt['print_gap']
        self.save_model = opt['save_model']
        self.evaluator = Evaluator(opt)
        self.model_type = model_type
        # init the loss and optimizer
        self.model = model
        self.criterion = nn.NLLLoss()
        learning_rate = opt['learning_rate']
        regu_strength = opt['regu_strength']
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.85, 0.98), eps=1e-09, weight_decay=regu_strength)

    def train(self, train_data, validation_data, model_manager):
        best_accuracy = 0
        start = time.time()
        for epoch in xrange(self.n_epoch):
            self.train_one_epoch(train_data, epoch)
            valid_accuracy = self.validate_one_epoch(validation_data, epoch)
            if valid_accuracy >= best_accuracy:
                best_accuracy = valid_accuracy
                model_manager.save_model(self.model, self.model_type)
        end = time.time()
        return end - start

    def train_one_epoch(self, train_data, epoch):
        self.model.train()
        avg_train_loss = 0.0
        train_size = 0
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
            train_size += labels.size()[0]
            avg_train_loss += loss.data[0]
        avg_train_loss /= train_size
        self.write_train_loss(epoch, avg_train_loss)

    def write_train_loss(self, epoch, avg_train_loss):
        loss_info = '%20s [%d]  training loss: %.4f' % \
                    (self.model_type, epoch + 1, avg_train_loss)
        print loss_info
        append_to_file(self.train_log_file, loss_info)

    # validate the model after each epoch, return the accuracy
    def validate_one_epoch(self, validation_data, epoch):
        accuracy, avg_valid_loss = self.evaluator.eval_accuracy_and_loss(self.model, validation_data)
        validation_info = '%20s [%d]  validation accuracy: %.4f; loss: %.4f' % \
                          (self.model_type, epoch + 1, accuracy, avg_valid_loss)
        metrics = self.evaluator.eval(self.model, self.model_type, validation_data)
        full_validation_info = format_list_to_string(['validation', metrics[:-1]], '\t')
        accuracy = metrics[0]
        print validation_info
        append_to_file(self.valid_log_file, full_validation_info)
        return accuracy

