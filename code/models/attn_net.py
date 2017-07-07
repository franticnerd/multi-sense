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
        # print self.attn
        # attn_weights = torch.mm(embeds, self.attn)
        attn_weights = self.attn(embeds)
        attn_weights = self.attn_softmax(attn_weights.transpose(0, 1))
        out = torch.mm(attn_weights, embeds)
        # out = torch.mean(embeds, dim=0)
        out = self.linear(out)
        log_probs = F.log_softmax(out)
        return log_probs