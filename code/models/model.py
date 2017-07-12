import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Recon(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_vocab_size):
        super(Recon, self).__init__()
        # self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, output_vocab_size)
    def forward(self, inputs):
        embeds = self.embedding(inputs)
        out = torch.mean(embeds, dim=0)
        out = self.linear(out)
        log_probs = F.log_softmax(out)
        return log_probs


class ReconNS(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_vocab_size):
        super(ReconNS, self).__init__()
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
    def get_embedding(self, idx):
        return self.embedding.weight[idx]
    def calc_similarities(self, query_embedding):
        return torch.mm(query_embedding.view(1,-1), self.embedding.weight.transpose(0, 1))
        # ret = []
        # for i in xrange(self.embedding.weight.size()[0]):
        #     embedding = self.embedding.weight[i]
        #     norm_prod = embedding.norm() * query_embedding.norm()
        #     denominator = np.max([1e-8, norm_prod.data[0]])
        #     similarity = np.dot(query_embedding.data.tolist(), embedding.data.tolist()) / denominator
        #     ret.append(similarity)
        # return Variable(torch.Tensor(ret)).view(1, -1)


class BilinearSenseNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_vocab_size, n_sense):
        super(BilinearSenseNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, output_vocab_size)
        self.attention_linear = nn.Linear(embedding_dim, embedding_dim)
        self.n_sense = n_sense
    def forward(self, inputs):
        n_words = inputs.size()[0] / self.n_sense
        embeds = self.embedding(inputs)
        context_vec = torch.mean(embeds, dim=0)
        context_vec = self.attention_linear(context_vec).transpose(0, 1)
        similarity_vec = torch.mm(embeds, context_vec)
        attn_weights = similarity_vec.view(-1, self.n_sense)
        attn_weights = F.softmax(attn_weights).view(1, -1) / n_words
        out = torch.mm(attn_weights, embeds)
        # out = torch.mean(embeds, dim=0)
        out = self.linear(out)
        log_probs = F.log_softmax(out)
        return log_probs


class BidirectionSenseNet(nn.Module):
    def __init__(self, x_vocab_size, embedding_dim, y_vocab_size, n_sense):
        super(BidirectionSenseNet, self).__init__()
        self.x_vocab_size = x_vocab_size
        self.y_vocab_size = y_vocab_size * n_sense
        self.x_embedder = nn.Embedding(x_vocab_size, embedding_dim)
        self.y_embedder = nn.Embedding(self.y_vocab_size, embedding_dim)
        self.n_sense = n_sense
        self.embedding_dim = embedding_dim
    def forward(self, x_input):
        # number of words
        n_words = x_input.size()[0] / self.n_sense
        x_embeds = self.x_embedder(x_input)
        context_vec = torch.mean(x_embeds, dim=0).transpose(0, 1)
        similarity_vec = torch.mm(x_embeds, context_vec)
        attn_weights = similarity_vec.view(-1, self.n_sense)
        attn_weights = F.softmax(attn_weights).view(1, -1) / n_words
        # use attention to get hidden state
        hidden = torch.mm(attn_weights, x_embeds).transpose(0, 1)
        # predict over y
        y_labels = torch.linspace(0, self.y_vocab_size - 1, self.y_vocab_size).long()
        y_labels = Variable(y_labels)
        y_embeds = self.y_embedder(y_labels)
        similarity_vec = torch.mm(y_embeds, context_vec)
        attn_weights = similarity_vec.view(-1, self.n_sense)
        attn_weights = F.softmax(attn_weights).view(1, -1)
        n_y_words = self.y_vocab_size / self.n_sense
        y_mean_vecs = torch.bmm(attn_weights.view(n_y_words , 1, self.n_sense),
                                y_embeds.view(n_y_words, self.n_sense, self.embedding_dim)).squeeze(1)
        # output
        out = torch.mm(y_mean_vecs, hidden).transpose(0, 1)
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
