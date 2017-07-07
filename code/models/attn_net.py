import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnNet(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_vocab_size):
        super(AttnNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, output_vocab_size)
        # self.attn = nn.Parameter(torch.Tensor(embedding_dim, 1))
        self.attn = nn.Linear(embedding_dim, 1)
        self.attn_softmax = nn.Softmax()


    def forward(self, inputs):
        embeds = self.embedding(inputs)
        # self.print_attn_parameters()
        # attn_weights = torch.mm(embeds, self.attn)
        attn_weights = self.attn(embeds)
        attn_weights = self.attn_softmax(attn_weights.transpose(0, 1))
        out = torch.mm(attn_weights, embeds)
        # out = torch.mean(embeds, dim=0)
        out = self.linear(out)
        log_probs = F.log_softmax(out)
        return log_probs


    def print_attn_parameters(self):
        for p in self.attn.parameters():
            print p



class SenseNet(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_vocab_size, n_sense):
        super(SenseNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, output_vocab_size)
        self.n_sense = n_sense


    def forward(self, inputs):
        n_words = inputs.size()[0] / self.n_sense
        embeds = self.embedding(inputs)
        context_vec = torch.mean(embeds, dim=0).transpose(0, 1)
        similarity_vec = torch.mm(embeds, context_vec)
        attn_weights = similarity_vec.view(-1, self.n_sense)
        attn_weights = F.softmax(attn_weights).view(1, -1) / n_words
        out = torch.mm(attn_weights, embeds)
        # out = torch.mean(embeds, dim=0)
        out = self.linear(out)
        log_probs = F.log_softmax(out)
        return log_probs
