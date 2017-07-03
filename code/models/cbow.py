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
