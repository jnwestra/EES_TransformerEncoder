import argparse
import json
import os
from os.path import join, exists
import pickle as pkl
import random #
from time import time
from datetime import timedelta

from cytoolz import compose

import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from encoder import SummarizerEncoder
from model.util import sequence_loss
from decoding import Extractor, DecodeDataset
from decoding import sort_ckpt, get_n_ext
from evaluate import eval_rouge

from utils import PAD, UNK
from utils import make_vocab, make_embedding

from data.data import ImgDmDataset
from data.batcher import tokenize
from data.batcher import coll_fn_extract
from data.batcher import prepro_fn_extract
from data.batcher import convert_batch_extract_ptr
from data.batcher import batchify_fn_extract_ptr
from data.batcher import BucketedGenerater

import warnings
warnings.filterwarnings("ignore", category=Warning)

BUCKET_SIZE = 6400

DATA_DIR = './IMGDM'

class ExtractDataset(ImgDmDataset):
    """ article sentences -> extraction indices
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        
        if type(js_data) is tuple: # If this is a previous sample
            return js_data
        
        art_sents, extracts = js_data['article'], js_data['label']
        return art_sents, extracts

def test(args, split):
    ext_dir = args.path
    ckpts = sort_ckpt(ext_dir)

    #setup loader
    def coll(batch):
        articles = list(filter(bool, batch))
        return articles
    dataset = DecodeDataset(split)

    n_data = len(dataset)
    loader = DataLoader(dataset, batch_size=args.batch,
        shuffle=False, collate_fn=coll)

    print('dataset length', n_data)

    # decode and eval top 5 models
    if not os.path.exists(join(args.path, 'decode')):
        os.mkdir(join(args.path, 'decode'))
        
    if not os.path.exists(join(args.path, 'ROUGE')):
        os.mkdir(join(args.path, 'ROUGE'))
        
    for i in range(min(5, len(ckpts))):
        print('Start loading checkpoint {} !'.format(ckpts[i]))
        cur_ckpt = torch.load(
                   join(ext_dir, 'ckpt/{}'.format(ckpts[i]))
        )['state_dict']
        extractor, _ = Extractor(ext_dir, cur_ckpt, cuda=args.cuda)
        save_path = join(args.path, 'decode/{}'.format(ckpts[i]))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # decoding
        ext_list = []
        cur_idx = 0
        start = time()
        with torch.no_grad():
            for raw_article_batch in loader:
                tokenized_article_batch = map(tokenize(None, args), raw_article_batch)
                for raw_art_sents in tokenized_article_batch:
                    ext_idx, _ = extractor(raw_art_sents)
                    ext_list.append(ext_idx)
                    cur_idx += 1
                    print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                          cur_idx, n_data, cur_idx/n_data*100, timedelta(seconds=int(time()-start))
                    ), end='')
        print()

        # write files
        for file_idx, ext_ids in enumerate(ext_list):
            dec = []
            data_path = join(DATA_DIR, '{}/{}.json'.format(split, file_idx))
            with open(data_path) as f:
                data = json.loads(f.read())
            n_ext = 3
            n_ext = min(n_ext, len(data['article']))
            for j in range(n_ext):
                sent_idx = ext_ids[j]
                dec.append(data['article'][sent_idx])
            with open(join(save_path, '{}.dec'.format(file_idx)), 'w') as f:
                for sent in dec:
                    print(sent, file=f)
        
        # evaluate current model
        print('Starting evaluating ROUGE !')
        dec_path = save_path
        ref_path = join(DATA_DIR, 'refs/{}'.format(split))
        print("eval_rouge")
        ROUGE = eval_rouge(dec_path, ref_path)
        print(ROUGE)
        with open(join(args.path, 'ROUGE/{}.txt'.format(ckpts[i])), 'w') as f:
            print(ROUGE, file=f)

def trained_encoder(args, split):
    result_path = args.result_path

    def coll(batch):
            articles = list(filter(bool, batch))
            return articles
    dataset = DecodeDataset(split)
    n_data = len(dataset)

    loader = DataLoader(dataset, batch_size=args.batch,
                        shuffle=False, num_workers=4, collate_fn=coll)

    ckpt_filename = join(result_path, 'ckpt', args.ckpt_name)
    ckpt = torch.load(ckpt_filename)['state_dict']

    extractor = Extractor(result_path, ckpt, cuda=args.cuda)

    enc_list = []
    cur_idx = 0
    start = time()
    with torch.no_grad():
    for raw_article_batch in loader:
        tokenized_article_batch = map(tokenize(None, args.emb_type), raw_article_batch)
        for raw_art_sents in tokenized_article_batch:
        _, enc_out = extractor(raw_art_sents)
        enc_list.append(enc_out)
        cur_idx += 1
        print('{}/{} ({:.2f}%) encoded in {} seconds\r'.format(
                cur_idx, n_data, cur_idx/n_data*100, timedelta(seconds=int(time()-start))
        ), end='')
    print(enc_list[0].size())
    return enc_list

class argWrapper(object):
  def __init__(self,
               ckpt_name,
               result_path='./result',
               batch=32,
               cuda=torch.cuda.is_available(),
               encoder_layer=12,
               encoder_hidden=512):
    self.ckpt_name = ckpt_name
    self.result_path = result_path
    self.batch = batch
    self.cuda = cuda
    self.encoder_layer = 12
    self.encoder_hidden = 512

if __name__ == '__main__':
    args = argWrapper('ckpt-0.313407-3000')

    enc_list = trained_encoder(args, 'test')




