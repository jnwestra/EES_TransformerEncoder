""" decoding utilities"""
import json
import re
import os
import torch
from os.path import join
import pickle as pkl
from itertools import starmap

from cytoolz import curry

from utils import PAD, UNK, START, END
from extract import Summarizer
#from model.rl import ActorCritic
from data.batcher import conver2id, pad_batch_tensorize
from data.data import ImgDmDataset, list_data


class DecodeDataset(ImgDmDataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, path):
        super().__init__(path)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents = js_data['article']
        return art_sents

class DecodeLabels(ImgDmDataset):
    def __init__(self, path):
        super().__init__(path)
    
    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_label = js_data['label']
        num_sents = len(js_data['article'])
        return art_label, num_sents

def make_html_safe(s):
    """Rouge use html, has to make output html safe"""
    return s.replace("<", "&lt;").replace(">", "&gt;")


def sort_ckpt(model_dir, reverse=False):
    """ reverse=False->loss, reverse=True->reward """
    ckpts = os.listdir(join(model_dir, 'ckpt'))
    ckpt_matcher = re.compile('^ckpt-.*-[0-9]*')
    ckpts = sorted([c for c in ckpts if ckpt_matcher.match(c)],
                   key=lambda c: float(c.split('-')[1]), reverse=reverse)
    return ckpts

def get_n_ext(split, idx):
    path = join(DATASET_DIR, '{}/{}.json'.format(split, idx))
    with open(path) as f:
        data = json.loads(f.read())
    if data['source'] == 'CNN':
        return 2
    else:
        return 3

class Decoder(object):
    def __init__(self, args, ckpt, data_root, max_ext=6):
        extractor = Summarizer(args.emb_dim, args.vocab_size, args.conv_hidden,
                                args.encoder_hidden, args.encoder_layer)
        extractor.load_state_dict(ckpt)

        word2id = pkl.load(open(join(data_root, 'vocab.pkl'), 'rb'))
        self._word2id = word2id

        self._device = torch.device('cuda' if args.cuda else 'cpu')
        self._net = extractor.to(self._device)
        ## self._id2word = {i: w for w, i in word2id.items()}
        self._max_ext = max_ext

    def __call__(self, raw_article_sents):
        self._net.eval()
        n_art = len(raw_article_sents)

        articles = conver2id(UNK, self._word2id, raw_article_sents)
        article = pad_batch_tensorize(articles, PAD, cuda=False).to(self._device)
        
        indices = self._net.extract([article], k=min(n_art, self._max_ext))
        return indices


class ArticleBatcher(object):
    def __init__(self, word2id, cuda=True):
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._word2id = word2id
        self._device = torch.device('cuda' if cuda else 'cpu')

    def __call__(self, raw_article_sents):
        article_sents = conver2id(UNK, self._word2id, raw_article_sents)
        article = pad_batch_tensorize(article_sents, PAD, cuda=False
                                     ).to(self._device)
        return article

