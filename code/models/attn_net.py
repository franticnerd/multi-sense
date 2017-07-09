import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_vocab_size):
        super(AttnNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, output_vocab_size)
        # self.attn = nn.Parameter(torch.randn(embedding_dim, 1))
        self.attn = nn.Linear(embedding_dim, 1)
        self.attn_softmax = nn.Softmax()
    def forward(self, inputs):
        embeds = self.embedding(inputs)  # n_words * dim
        # self.print_attn_parameters()
        # attn_weights = torch.mm(embeds, self.attn)
        attn_weights = self.attn(embeds)  # 1 * n_words
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


class AttnSenseNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_vocab_size, n_sense):
        super(AttnSenseNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(embedding_dim, output_vocab_size)
        self.n_sense = n_sense
        self.attn = nn.Linear(embedding_dim, 1)
        self.attn_softmax = nn.Softmax()
    def forward(self, inputs):
        n_words = inputs.size()[0] / self.n_sense
        embeds = self.embedding(inputs)
        # calc the context vector
        word_mean_vecs = embeds.view(n_words, self.n_sense, -1)  # n_words * n_sense *  dim
        word_mean_vecs = torch.mean(word_mean_vecs, dim=1).squeeze(1)  # n_words * dim
        word_importance = self.attn(word_mean_vecs)
        word_importance = self.attn_softmax(word_importance.transpose(0, 1)) # 1*n_word
        context_vec = torch.mm(word_importance, word_mean_vecs).transpose(0, 1)
        # compute attention weights
        similarity_vec = torch.mm(embeds, context_vec)
        attn_weights = similarity_vec.view(-1, self.n_sense)
        attn_weights = F.softmax(attn_weights).view(1, -1) / n_words
        # output vector
        out = torch.mm(attn_weights, embeds)
        out = self.linear(out)
        log_probs = F.log_softmax(out)
        return log_probs

class CompAttnSenseNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_vocab_size, n_sense):
        super(CompAttnSenseNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(embedding_dim, output_vocab_size)
        self.n_sense = n_sense
        self.attn = nn.Linear(embedding_dim, 1)
        self.attn_softmax = nn.Softmax()
    def forward(self, inputs):
        n_words = inputs.size()[0] / self.n_sense
        embeds = self.embedding(inputs)
        # compute attentions over the senses and then the attentional word mean vectors
        mean_word_embedding = torch.mean(embeds, dim=0).transpose(0, 1)
        sense_similarity_vec = torch.mm(embeds, mean_word_embedding)  # the similarity between word embedding and mean embedding
        word_attn_weights = sense_similarity_vec.view(n_words, self.n_sense)  # the attention over each sense for a word
        word_attn_weights = F.softmax(word_attn_weights).view(1, -1) # n_words * n_sense
        word_mean_vecs = torch.bmm(word_attn_weights.view(n_words, 1, self.n_sense),
                                   embeds.view(n_words, self.n_sense, self.embedding_dim)).squeeze(1) # n_words * dim
        # calc the context vector
        word_importance = self.attn(word_mean_vecs)
        word_importance = self.attn_softmax(word_importance.transpose(0, 1)) # 1*n_word
        context_vec = torch.mm(word_importance, word_mean_vecs).transpose(0, 1)
        # compute attention weights over the words
        similarity_vec = torch.mm(embeds, context_vec)
        attn_weights = similarity_vec.view(-1, self.n_sense)
        attn_weights = F.softmax(attn_weights).view(1, -1) / n_words
        # output vector
        out = torch.mm(attn_weights, embeds)
        out = self.linear(out)
        log_probs = F.log_softmax(out)
        return log_probs
