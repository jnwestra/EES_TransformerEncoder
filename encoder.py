import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from model.TransformerEncoder import TransformerEncoder

import numpy as np
import math
from math import sqrt

from transformer.Models import get_sinusoid_encoding_table

INI = 1e-2
MAX_ARTICLE_LEN = 512


# Copy of Summarizer class from extract.py, but with only W2V, Transformer, and SL
class SummarizerEncoder(nn.Module):
    def __init__(self, emb_dim, vocab_size,
                conv_hidden, encoder_hidden, encoder_layer,
                isTrain=True, n_hop=1, dropout=0.0):
        super().__init__()

        self._sent_enc = ConvSentEncoder(vocab_size, emb_dim,
            conv_hidden, dropout)

        # Transformer Encoder
        enc_out_dim = encoder_hidden
        self._art_enc = TransformerEncoder(
            3*conv_hidden, encoder_hidden, encoder_layer)
        
        self._emb_w = nn.Linear(3*conv_hidden, encoder_hidden)
        self.sent_pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(1000, enc_out_dim, padding_idx=0),
            freeze=True)
        
        # SL decoder
        self._ws = nn.Linear(enc_out_dim, 2)


    def forward(self, article_sents, sent_nums):
        enc_out = self._encode(article_sents, sent_nums)

        bs, seq_len, d = enc_out.size()
        output = self._ws(enc_out)
        assert output.size() == (bs,seq_len,2)

        return output, enc_out

    def _encode(self, article_sents, sent_nums):
        hidden_size = self._art_enc.input_size

        if sent_nums is None:
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