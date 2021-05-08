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
from encoder import SummarizerEncoder
#from model.rl import ActorCritic
from data.batcher import conver2id, pad_batch_tensorize
from data.data import ImgDmDataset


class DecodeDataset(ImgDmDataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split, path):
        assert split in ['val', 'test']
        super().__init__(split, path)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents = js_data['article']
        return art_sents


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

class Extractor(object):
    def __init__(self, ext_dir, ext_ckpt, max_ext=6, cuda=True):
        ext_meta = json.load(open(join(ext_dir, 'meta.json')))
        ext_args = ext_meta['model_args']
        _, extractor = SummarizerEncoder(**ext_args)
        extractor.load_state_dict(ext_ckpt)

        word2id = pkl.load(open(join(ext_dir, 'vocab.pkl'), 'rb'))
        self._word2id = word2id

        self._device = torch.device('cuda' if cuda else 'cpu')
        self._net = extractor.to(self._device)
        ## self._id2word = {i: w for w, i in word2id.items()}
        self._max_ext = max_ext

    def __call__(self, raw_article_sents):
        self._net.eval()
        n_art = len(raw_article_sents)

        articles = conver2id(UNK, self._word2id, raw_article_sents)
        article = pad_batch_tensorize(articles, PAD, cuda=False
                                     ).to(self._device)
        indices = self._net.extract([article], k=min(n_art, self._max_ext))
        return indices


class ArticleBatcher(object):
    def __init__(self, word2id, cuda=True):
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._word2id = word2id
        self._device = torch.device('cuda' if cuda else 'cpu')

    def __call__(self, raw_article_sents):
        articles = conver2id(UNK, self._word2id, raw_article_sents)
        article = pad_batch_tensorize(articles, PAD, cuda=False
                                     ).to(self._device)
        return article

