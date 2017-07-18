import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import constants


class Recon(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_vocab_size):
        super(Recon, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=constants.PAD)
        self.linear = nn.Linear(embedding_dim, output_vocab_size)
    def forward(self, inputs, length_weights, word_attn_mask):
        hidden = self.calc_hidden(inputs, length_weights, word_attn_mask)
        out = self.linear(hidden)
        log_probs = F.log_softmax(out)
        return log_probs
    def calc_hidden(self, inputs, length_weights, word_attn_mask):
        embeds = self.embedding(inputs)
        out = torch.sum(embeds, dim=1)
        out = torch.bmm(length_weights, out).squeeze(1)
        return out


class ReconNS(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_vocab_size):
        super(ReconNS, self).__init__()
        self.embeder_x = nn.Embedding(vocab_size, embedding_dim)
        self.embeder_y = nn.Embedding(output_vocab_size, embedding_dim)
    def forward(self, inputs, labels):
        x_embeds = self.embeder_x(inputs)
        x_mean = torch.mean(x_embeds, dim=0)
        y_embeds = self.embeder_y(labels).transpose(0, 1)
        similarities = torch.mm(x_mean, y_embeds)
        prob = F.sigmoid(similarities)
        return prob

class AttnNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_vocab_size):
        super(AttnNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=constants.PAD)
        self.linear = nn.Linear(embedding_dim, output_vocab_size)
        self.attn = nn.Linear(embedding_dim, 1)
        self.attn_softmax = nn.Softmax()
    def forward(self, inputs, length_weights, word_attn_mask):
        hidden = self.calc_hidden(inputs, length_weights, word_attn_mask)
        out = self.linear(hidden)
        log_probs = F.log_softmax(out)
        return log_probs
    def calc_hidden(self, inputs, length_weights, word_attn_mask):
        embeds = self.embedding(inputs)
        mb_size, max_len, embedding_dim = embeds.size()
        embeds = embeds.view(-1, embedding_dim)
        attn_weights = self.attn(embeds).view(mb_size, max_len)
        attn_mask = self.get_attn_mask(inputs)
        attn_weights.data.masked_fill_(attn_mask, -float('inf'))
        attn_weights = self.attn_softmax(attn_weights).unsqueeze(1)
        embeds = embeds.view(mb_size, max_len, embedding_dim)
        out = torch.bmm(attn_weights, embeds).squeeze(1)
        return out
    # get the masks for computing mini-batch attention
    def get_attn_mask(self, inputs):
        pad_attn_mask = inputs.data.eq(constants.PAD)   # mb_size x max_len
        return pad_attn_mask

class SenseNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_vocab_size, n_sense):
        super(SenseNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=constants.PAD)
        self.linear = nn.Linear(embedding_dim, output_vocab_size)
        self.n_sense = n_sense
        self.embedding_dim = embedding_dim
    def forward(self, inputs, length_weights, word_attn_mask):
        hidden = self.calc_hidden(inputs, length_weights, word_attn_mask)
        out = self.linear(hidden)
        log_probs = F.log_softmax(out)
        return log_probs
    def calc_hidden(self, inputs, length_weights, word_attn_mask):
        mb_size, max_len = inputs.size()[0], inputs.size()[1] / self.n_sense
        embeds = self.embedding(inputs) # mb_size * (max_len * embedding_dim)
        # compute the context vector
        embeds = embeds.view(mb_size, max_len, self.n_sense, self.embedding_dim)
        embed_mean = torch.mean(embeds, dim=2).squeeze(2)  # mb_size * max_len * embedding_dim
        embed_mean = torch.sum(embed_mean, dim=1)  # mb_size * 1 * embedding_dim
        context_vec = torch.bmm(length_weights, embed_mean)  # mb_size * 1 * embedding_dim
        # compute the similarities
        embeds = embeds.view(mb_size, -1, self.embedding_dim)  # mb_size * (max_len * n_sense) * embedding_dim
        context_vec = context_vec.transpose(1, 2)  # mb_size * embedding_dim * 1
        similarity_vec = torch.bmm(embeds, context_vec)  # mb_size * (max_len * n_sense) * 1
        # get the attention weights over the senses
        attn_weights = similarity_vec.view(-1, self.n_sense)  # (mb_size * max_len) * n_sense
        attn_weights = F.softmax(attn_weights).view(mb_size, 1,  -1)  # mb_size * 1 * (max_len * n_sense)
        attn_weights = torch.bmm(length_weights, attn_weights).squeeze(1)  # scale by length, mb_size * (max_len * n_sense)
        attn_mask = self.get_attn_mask(inputs)
        attn_weights.data.masked_fill_(attn_mask, 0)  # mb_size * (max_len * n_sense)
        attn_weights = attn_weights.view(mb_size, 1, -1)  # mb_size * 1 * (max_len * n_sense)
        # now use the attention to get the hidden state
        hidden = torch.bmm(attn_weights, embeds).squeeze(1)
        return hidden
    # get the masks for computing mini-batch attention
    def get_attn_mask(self, inputs):
        pad_attn_mask = inputs.data.eq(constants.PAD)   # mb_size x max_len
        return pad_attn_mask



class BilinearSenseNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_vocab_size, n_sense):
        super(BilinearSenseNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=constants.PAD)
        self.linear = nn.Linear(embedding_dim, output_vocab_size)
        self.attention_linear = nn.Linear(embedding_dim, embedding_dim)
        self.n_sense = n_sense
        self.embedding_dim = embedding_dim
    def forward(self, inputs, length_weights, word_attn_mask):
        hidden = self.calc_hidden(inputs, length_weights, word_attn_mask)
        out = self.linear(hidden)
        log_probs = F.log_softmax(out)
        return log_probs
    def calc_hidden(self, inputs, length_weights, word_attn_mask):
        mb_size, max_len = inputs.size()[0], inputs.size()[1] / self.n_sense
        embeds = self.embedding(inputs) # mb_size * (max_len * embedding_dim)
        # compute the context vector
        embeds = embeds.view(mb_size, max_len, self.n_sense, self.embedding_dim)
        embed_mean = torch.mean(embeds, dim=2).squeeze(2)  # mb_size * max_len * embedding_dim
        embed_mean = torch.sum(embed_mean, dim=1)  # mb_size * 1 * embedding_dim
        context_vec = torch.bmm(length_weights, embed_mean).squeeze(1)  # mb_size * embedding_dim
        context_vec = self.attention_linear(context_vec).unsqueeze(1)  # mb_size * 1 * embedding_dim
        # compute the similarities
        embeds = embeds.view(mb_size, -1, self.embedding_dim)  # mb_size * (max_len * n_sense) * embedding_dim
        context_vec = context_vec.transpose(1, 2)  # mb_size * embedding_dim * 1
        similarity_vec = torch.bmm(embeds, context_vec)  # mb_size * (max_len * n_sense) * 1
        # get the attention weights over the senses
        attn_weights = similarity_vec.view(-1, self.n_sense)  # (mb_size * max_len) * n_sense
        attn_weights = F.softmax(attn_weights).view(mb_size, 1,  -1)  # mb_size * 1 * (max_len * n_sense)
        attn_weights = torch.bmm(length_weights, attn_weights).squeeze(1)  # scale by length, mb_size * (max_len * n_sense)
        attn_mask = self.get_attn_mask(inputs)
        attn_weights.data.masked_fill_(attn_mask, 0)  # mb_size * (max_len * n_sense)
        attn_weights = attn_weights.view(mb_size, 1, -1)  # mb_size * 1 * (max_len * n_sense)
        # now use the attention to get the hidden state
        hidden = torch.bmm(attn_weights, embeds).squeeze(1)
        return hidden
    # get the masks for computing mini-batch attention
    def get_attn_mask(self, inputs):
        pad_attn_mask = inputs.data.eq(constants.PAD)   # mb_size x max_len
        return pad_attn_mask


class BidirectionSenseNet(nn.Module):
    def __init__(self, x_vocab_size, embedding_dim, y_vocab_size, n_sense):
        super(BidirectionSenseNet, self).__init__()
        self.x_vocab_size = x_vocab_size
        self.y_vocab_size = y_vocab_size * n_sense
        self.x_embedder = nn.Embedding(x_vocab_size, embedding_dim, padding_idx=constants.PAD)
        self.y_embedder = nn.Embedding(self.y_vocab_size, embedding_dim)
        self.n_sense = n_sense
        self.embedding_dim = embedding_dim
    def forward(self, inputs, length_weights, word_attn_mask):
        mb_size, max_len = inputs.size()[0], inputs.size()[1] / self.n_sense
        embeds = self.x_embedder(inputs) # mb_size * (max_len * embedding_dim)
        # compute the context vector
        embeds = embeds.view(mb_size, max_len, self.n_sense, self.embedding_dim)
        embed_mean = torch.mean(embeds, dim=2).squeeze(2)  # mb_size * max_len * embedding_dim
        embed_mean = torch.sum(embed_mean, dim=1)  # mb_size * 1 * embedding_dim
        context_vec = torch.bmm(length_weights, embed_mean)  # mb_size * 1 * embedding_dim
        # compute the similarities
        embeds = embeds.view(mb_size, -1, self.embedding_dim)  # mb_size * (max_len * n_sense) * embedding_dim
        context_vec = context_vec.transpose(1, 2)  # mb_size * embedding_dim * 1
        similarity_vec = torch.bmm(embeds, context_vec)  # mb_size * (max_len * n_sense) * 1
        # get the attention weights over the senses
        attn_weights = similarity_vec.view(-1, self.n_sense)  # (mb_size * max_len) * n_sense
        attn_weights = F.softmax(attn_weights).view(mb_size, 1,  -1)  # mb_size * 1 * (max_len * n_sense)
        attn_weights = torch.bmm(length_weights, attn_weights).squeeze(1)  # scale by length, mb_size * (max_len * n_sense)
        attn_mask = self.get_attn_mask(inputs)
        attn_weights.data.masked_fill_(attn_mask, 0)  # mb_size * (max_len * n_sense)
        attn_weights = attn_weights.view(mb_size, 1, -1)  # mb_size * 1 * (max_len * n_sense)
        # now use the attention to get the hidden state
        hidden = torch.bmm(attn_weights, embeds)  # mb_size * 1 * dim
        # predict over y
        y_labels = torch.linspace(0, self.y_vocab_size - 1, self.y_vocab_size).long()
        y_labels = Variable(y_labels.view(1, self.y_vocab_size).expand(mb_size, self.y_vocab_size))
        y_embeds = self.y_embedder(y_labels)  # mb_size * y_vocab_size * dim
        similarity_vec = torch.bmm(y_embeds, context_vec)  # mb_size * y_vocab_size (namely vocab * n_sense)
        # compute the attention vectors over y senses
        attn_weights = similarity_vec.view(-1, self.n_sense)
        attn_weights = F.softmax(attn_weights).view(mb_size, -1)  # mb_size * y_vocab_size
        n_y_words = self.y_vocab_size / self.n_sense
        attn_weights = attn_weights.view(mb_size * n_y_words, 1, self.n_sense)
        y_embeds = y_embeds.view(mb_size * n_y_words, self.n_sense, self.embedding_dim)
        y_mean_vecs = torch.bmm(attn_weights, y_embeds).squeeze(1).view(mb_size, n_y_words, self.embedding_dim)  # mb_size * n_y_word * dim
        # output
        y_mean_vecs = y_mean_vecs.transpose(1, 2)  # mb_size * dim * n_y_word
        out = torch.bmm(hidden, y_mean_vecs).squeeze(1)  # mb_size * n_y_word
        log_probs = F.log_softmax(out)
        return log_probs
    def get_attn_mask(self, inputs):
        pad_attn_mask = inputs.data.eq(constants.PAD)   # mb_size x max_len
        return pad_attn_mask



class AttnSenseNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_vocab_size, n_sense):
        super(AttnSenseNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=constants.PAD)
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(embedding_dim, output_vocab_size)
        self.n_sense = n_sense
        self.attn = nn.Linear(embedding_dim, 1)
        self.attn_softmax = nn.Softmax()
    def forward(self, inputs, length_weights, word_attn_mask):
        hidden = self.calc_hidden(inputs, length_weights, word_attn_mask)
        out = self.linear(hidden)
        log_probs = F.log_softmax(out)
        return log_probs
    def calc_hidden(self, inputs, length_weights, word_attn_mask):
        mb_size, max_len = inputs.size()[0], inputs.size()[1] / self.n_sense
        embeds = self.embedding(inputs) # mb_size * (max_len * embedding_dim)
        # compute the context vector
        embeds = embeds.view(mb_size, max_len, self.n_sense, self.embedding_dim) # take the mean over senses
        embed_mean = torch.mean(embeds, dim=2).squeeze(2)  # mb_size * max_len * embedding_dim
        embed_mean = embed_mean.view(-1, self.embedding_dim)  # (mb_size * max_len) * embedding_dim
        word_importance = self.attn(embed_mean).view(mb_size, max_len)  # mb_size * max_len
        word_importance.data.masked_fill_(word_attn_mask, -float('inf'))
        word_importance = self.attn_softmax(word_importance).unsqueeze(1)  # mb_size * 1 * max_len
        embed_mean = embed_mean.view(mb_size, max_len, self.embedding_dim)  # mb_size * max_len * embedding_dim
        context_vec = torch.bmm(word_importance, embed_mean)  # mb_size * 1 * embedding_dim
        # compute the similarities
        embeds = embeds.view(mb_size, -1, self.embedding_dim)  # mb_size * (max_len * n_sense) * embedding_dim
        context_vec = context_vec.transpose(1, 2)  # mb_size * embedding_dim * 1
        similarity_vec = torch.bmm(embeds, context_vec)  # mb_size * (max_len * n_sense) * 1
        # get the attention weights over the senses
        attn_weights = similarity_vec.view(-1, self.n_sense)  # (mb_size * max_len) * n_sense
        attn_weights = F.softmax(attn_weights).view(mb_size, 1,  -1)  # mb_size * 1 * (max_len * n_sense)
        attn_weights = torch.bmm(length_weights, attn_weights).squeeze(1)  # scale by length, mb_size * (max_len * n_sense)
        attn_mask = self.get_attn_mask(inputs)
        attn_weights.data.masked_fill_(attn_mask, 0)  # mb_size * (max_len * n_sense)
        attn_weights = attn_weights.view(mb_size, 1, -1)  # mb_size * 1 * (max_len * n_sense)
        # now use the attention to get the hidden state
        out = torch.bmm(attn_weights, embeds).squeeze(1)
        return out
    # get the masks for computing mini-batch attention
    def get_attn_mask(self, inputs):
        pad_attn_mask = inputs.data.eq(constants.PAD)   # mb_size x max_len
        return pad_attn_mask



class CompAttnSenseNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_vocab_size, n_sense):
        super(CompAttnSenseNet, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=constants.PAD)
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(embedding_dim, output_vocab_size)
        self.n_sense = n_sense
        self.attn = nn.Linear(embedding_dim, 1)
        self.attn_softmax = nn.Softmax()
    def forward(self, inputs, length_weights, word_attn_mask):
        hidden = self.calc_hidden(inputs, length_weights, word_attn_mask)
        out = self.linear(hidden)
        log_probs = F.log_softmax(out)
        return log_probs
    def calc_hidden(self, inputs, length_weights, word_attn_mask):
        mb_size, max_len = inputs.size()[0], inputs.size()[1] / self.n_sense
        embeds = self.embedding(inputs) # mb_size * (max_len * sense) * embedding_dim
        # compute the attentional mean embeddings over senses
        global_mean = torch.sum(embeds, dim=1)  # mb_size * 1 * embedding_dim
        global_mean = torch.bmm(length_weights, global_mean).squeeze(1) / self.n_sense  # mean over all words and sense, mb_size * embedding_dim
        global_mean = global_mean.view(mb_size, self.embedding_dim, 1)
        sense_importance = torch.bmm(embeds, global_mean)  # similarity between global embedding and sense, mb_size * (max_len * sense) * 1
        sense_importance = sense_importance.view(-1, self.n_sense)  # (mb_size * max_len) * sense
        sense_importance = F.softmax(sense_importance).view(mb_size, 1, -1)  # mb_size * 1 * (max_len * sense)
        sense_importance = sense_importance.view(mb_size * max_len, 1, self.n_sense)  # (mb_size * max_len) * 1 * n_sense
        word_mean = torch.bmm(sense_importance, embeds.view(mb_size * max_len, self.n_sense, self.embedding_dim)).squeeze(1) # (mb_size * max_len) * dim
        # compute the attentional mean embeddings over words
        word_importance = self.attn(word_mean).view(mb_size, max_len)  # mb_size * max_len
        word_importance.data.masked_fill_(word_attn_mask, -float('inf'))
        word_importance = self.attn_softmax(word_importance).unsqueeze(1)  # mb_size * 1 * max_len
        word_mean = word_mean.view(mb_size, max_len, self.embedding_dim)  # mb_size * max_len * embedding_dim
        context_vec = torch.bmm(word_importance, word_mean)  # mb_size * 1 * embedding_dim
        # compute the similarities
        embeds = embeds.view(mb_size, -1, self.embedding_dim)  # mb_size * (max_len * n_sense) * embedding_dim
        context_vec = context_vec.transpose(1, 2)  # mb_size * embedding_dim * 1
        similarity_vec = torch.bmm(embeds, context_vec)  # mb_size * (max_len * n_sense) * 1
        # get the attention weights over the senses
        attn_mask = self.get_attn_mask(inputs)
        attn_weights = similarity_vec.view(-1, self.n_sense)  # (mb_size * max_len) * n_sense
        attn_weights = F.softmax(attn_weights).view(mb_size, 1,  -1)  # mb_size * 1 * (max_len * n_sense)
        attn_weights = torch.bmm(length_weights, attn_weights).squeeze(1)  # scale by length, mb_size * (max_len * n_sense)
        attn_weights.data.masked_fill_(attn_mask, 0)  # mb_size * (max_len * n_sense)
        attn_weights = attn_weights.view(mb_size, 1, -1)  # mb_size * 1 * (max_len * n_sense)
        # now use the attention to get the hidden state
        out = torch.bmm(attn_weights, embeds).squeeze(1)
        return out
    # get the masks for computing mini-batch attention
    def get_attn_mask(self, inputs):
        pad_attn_mask = inputs.data.eq(constants.PAD)   # mb_size x max_len
        return pad_attn_mask


