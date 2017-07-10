import torch
import torch.nn as nn
import torch.nn.functional as F


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_vocab_size):
        super(CBOW, self).__init__()
        # self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, output_vocab_size)
    def forward(self, inputs):
        embeds = self.embedding(inputs)
        out = torch.mean(embeds, dim=0)
        out = self.linear(out)
        log_probs = F.log_softmax(out)
        return log_probs


class CBOWNEG(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_vocab_size):
        super(CBOWNEG, self).__init__()
        # self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim)
        self.embeder_x = nn.Embedding(vocab_size, embedding_dim)
        self.embeder_y = nn.Embedding(output_vocab_size, embedding_dim)
    def forward(self, inputs, labels):
        x_embeds = self.embeder_x(inputs)
        # print self.embeder_x.weight[6]
        x_mean = torch.mean(x_embeds, dim=0)
        y_embeds = self.embeder_y(labels).transpose(0, 1)
        similarities = torch.mm(x_mean, y_embeds)
        prob = F.sigmoid(similarities)
        return prob
