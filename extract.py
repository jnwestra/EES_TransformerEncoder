import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from model.TransformerEncoder import TransformerEncoder

import os
from os.path import join

import numpy as np
import math
from math import sqrt

from utils import PAD, UNK, START, END
from data.batcher import conver2id, pad_batch_tensorize
import pickle as pkl

from transformer.Models import get_sinusoid_encoding_table

INI = 1e-2
MAX_ARTICLE_LEN = 512


# Copy of Summarizer class from original EES, but with only W2V, Transformer, and SL
class Summarizer(nn.Module):
    def __init__(self, emb_dim, vocab_size, conv_hidden, encoder_hidden,
        encoder_layer, isTrain=False, n_hop=1, dropout=0.0):

        super().__init__()

        self._sent_enc = ConvSentEncoder(
            vocab_size, emb_dim, conv_hidden, dropout, emb_type)

        # Sentence Encoder (Transformer)
        enc_out_dim = encoder_hidden
        self._art_enc = TransformerEncoder(
            3*conv_hidden, encoder_hidden, encoder_layer, decoder)
        
        self._emb_w = nn.Linear(3*conv_hidden, encoder_hidden)
        self.sent_pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(1000, enc_out_dim, padding_idx=0), freeze=True)

        # Decoder (SL)
        self._ws = nn.Linear(enc_out_dim, 2)
            

    def forward(self, article_sents, sent_nums, target):
        enc_out = self._encode(article_sents, sent_nums)

        bs, seq_len, d = enc_out.size()
        output = self._ws(enc_out)
        assert output.size() == (bs, seq_len, 2)

        return output

    def extract(self, article_sents, sent_nums=None, k=4):
        enc_out = self._encode(article_sents, sent_nums)
        
        seq_len = enc_out.size(1)
        output = self._ws(enc_out)
        assert output.size() == (1, seq_len, 2)
        _, indices = output[:, :, 1].sort(descending=True)
        extract = []
        for i in range(k):
            extract.append(indices[0][i].item())

        return extract

    def _encode(self, article_sents, sent_nums):
        
        hidden_size = self._art_enc.input_size

        if sent_nums is None:  # test-time excode only
            enc_sent = self._sent_enc(article_sents[0]).unsqueeze(0)
        else:
            max_n = max(sent_nums)
            enc_sents = [self._sent_enc(art_sent) for art_sent in article_sents]
            def zero(n, device):
                z = torch.zeros(n, hidden_size).to(device)
                return z
            enc_sent = torch.stack(
                [torch.cat([s, zero(max_n-n, s.get_device())], dim=0)
                   if n != max_n
                 else s
                 for s, n in zip(enc_sents, sent_nums)],
                dim=0
            )
        
        # Pass to Transformer Encoder
        batch_size, seq_len = enc_sent.size(0), enc_sent.size(1)

        # prepare mask
        if sent_nums != None:
            input_len = len_mask(sent_nums, enc_sent.get_device()).float() # [batch_size, seq_len]
        else:
            input_len = torch.ones(batch_size, seq_len).float().cuda()

        attn_mask = input_len.eq(0.0).unsqueeze(1).expand(batch_size, 
                    seq_len, seq_len).cuda() # [batch_size, seq_len, seq_len]
        non_pad_mask = input_len.unsqueeze(-1).cuda()  # [batch, seq_len, 1]

        # add postional embedding
        if sent_nums != None:
            sent_pos = torch.LongTensor([np.hstack((np.arange(1, doclen + 1), 
                        np.zeros(seq_len - doclen))) for doclen in sent_nums]).cuda()
        else:
            sent_pos = torch.LongTensor([np.arange(1, seq_len + 1)]).cuda()

        inputs = self._emb_w(enc_sent) + self.sent_pos_embed(sent_pos)

        assert attn_mask.size() == (batch_size, seq_len, seq_len)
        assert non_pad_mask.size() == (batch_size, seq_len, 1)
        
        output = self._art_enc(inputs, non_pad_mask, attn_mask)

        return output

    def _article_encode(self, article, device, pad_idx=0):
        sent_num, sent_len = article.size()
        tokens_id = [101] # [CLS]
        for i in range(sent_num):
            for j in range(sent_len):
                if article[i][j] != pad_idx:
                    tokens_id.append(article[i][j])
                else:
                    break
        tokens_id.append(102) # [SEP]
        input_mask = [1] * len(tokens_id)
        total_len = len(tokens_id) - 2
        while len(tokens_id) < MAX_ARTICLE_LEN:
            tokens_id.append(0)
            input_mask.append(0)

        assert len(tokens_id)  == MAX_ARTICLE_LEN
        assert len(input_mask) == MAX_ARTICLE_LEN

        input_ids = torch.LongTensor(tokens_id).unsqueeze(0).to(device)
        input_mask = torch.LongTensor(input_mask).unsqueeze(0).to(device)
        
        # concat last 4 layers
        out, _ = self._bert(input_ids, token_type_ids=None, attention_mask=input_mask)
        out = torch.cat([out[-1], out[-2], out[-3], out[-4]], dim=-1)

        assert out.size() == (1, MAX_ARTICLE_LEN, 4096)
        
        emb_out = self._bert_w(out).squeeze(0)
        emb_dim = emb_out.size(-1)

        emb_input = torch.zeros(sent_num, sent_len, emb_dim).to(device)
        cur_idx = 1 # after [CLS]
        for i in range(sent_num):
            for j in range(sent_len):
                if article[i][j] != pad_idx:
                    emb_input[i][j] = emb_out[cur_idx]
                    cur_idx += 1
                else:
                    break
        assert cur_idx - 1 == total_len

        cnn_out = self._sent_enc(emb_input)
        assert cnn_out.size() == (sent_num, 300) # 300 = 3 * conv_hidden

        return cnn_out

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)

class SummarizerEncoder(nn.Module):
    def __init__(self, emb_dim, vocab_size, conv_hidden, encoder_hidden,
        encoder_layer, data_root, isTrain=False, n_hop=1, dropout=0.0):

        super(SummarizerEncoder,self).__init__()

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        word2id = pkl.load(open(join(data_root, 'vocab.pkl'), 'rb'))
        self._word2id = word2id


        self._sent_enc = ConvSentEncoder(vocab_size, emb_dim,
            conv_hidden, dropout)

        # Transformer Encoder
        enc_out_dim = encoder_hidden
        self._art_enc = TransformerEncoder(
            3*conv_hidden, encoder_hidden, encoder_layer)
        
        self._emb_w = nn.Linear(3*conv_hidden, encoder_hidden)
        self.sent_pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(1000, enc_out_dim, padding_idx=0))

    def forward(self, raw_article_sents):
        article_sents = conver2id(UNK, self._word2id, raw_article_sents)
        article = pad_batch_tensorize(article_sents, PAD, cuda=False
                                     ).to(self._device)
        enc_out = self._encode([article])
        return enc_out

    def _encode(self, article_sents):
        hidden_size = self._art_enc.input_size
        enc_sents = [self._sent_enc(art_sent) for art_sent in article_sents]
        
        def zero(n, device):
            z = torch.zeros(n, hidden_size).to(device)
            return z

        N = len(enc_sents)
        
        sent_nums = list(range(N))

        enc_sent = torch.stack(
            [torch.cat([s, zero(N-n, s.get_device())], dim=0)
            if n != N
            else s
            for s, n in zip(enc_sents, sent_nums)],
            dim=0
        )

        batch_size, seq_len = enc_sent.size(0), enc_sent.size(1)

        # prepare mask
        if sent_nums != None:
            input_len = len_mask(sent_nums, enc_sent.get_device()).float() # [batch_size, seq_len]
        else:
            input_len = torch.ones(batch_size, seq_len).float().cuda()

        attn_mask = input_len.eq(0.0).unsqueeze(1).expand(batch_size, 
                    seq_len, seq_len).cuda() # [batch_size, seq_len, seq_len]
        non_pad_mask = input_len.unsqueeze(-1).cuda()  # [batch, seq_len, 1]

        # add postional embedding
        if sent_nums != None:
            sent_pos = torch.LongTensor([np.hstack((np.arange(1, doclen + 1), 
                        np.zeros(seq_len - doclen))) for doclen in sent_nums]).cuda()
        else:
            sent_pos = torch.LongTensor([np.arange(1, seq_len + 1)]).cuda()

        inputs = self._emb_w(enc_sent) + self.sent_pos_embed(sent_pos)

        assert attn_mask.size() == (batch_size, seq_len, seq_len)
        assert non_pad_mask.size() == (batch_size, seq_len, 1)
        
        output = self._art_enc(inputs, non_pad_mask, attn_mask)
        return output

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)



class ConvSentEncoder(nn.Module):
    """
    Convolutional word-level sentence encoder
    w/ max-over-time pooling, [3, 4, 5] kernel sizes, ReLU activation
    """
    def __init__(self, vocab_size, emb_dim, n_hidden, dropout):
        super().__init__()
        self._embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self._convs = nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, i)
                                     for i in range(3, 6)])
        self._dropout = dropout
        self._grad_handle = None

    def forward(self, input_):
        emb_input = self._embedding(input_)
        conv_in = F.dropout(emb_input.transpose(1, 2),
                            self._dropout, training=False)
        output = torch.cat([F.relu(conv(conv_in)).max(dim=2)[0]
                            for conv in self._convs], dim=1)
        return output

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)