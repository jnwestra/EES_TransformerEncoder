import json
import os
from os.path import join, exists
import random #
import time, datetime
from time import time
from datetime import timedelta

import torch
from torch.utils.data import DataLoader

from extract import SummarizerEncoder
from model.util import sequence_loss
from decoding import Decoder, DecodeDataset, DecodeLabels
from evaluate import eval_rouge

from data.data import ImgDmDataset, list_data
from data.batcher import tokenize

import warnings
warnings.filterwarnings("ignore", category=Warning)

import sys
sys.setrecursionlimit(10000)

DATA_DIR = 'IMGDM'

def test(args, split):
    
    # setup loader
    def coll(batch):
        articles = list(filter(bool, batch))
        return articles
    data_root= join(args.project_path,DATA_DIR)
    data_path = join(data_root,split)


    dataset = DecodeDataset(data_path)
    n_data = len(dataset)

    loader = DataLoader(dataset, batch_size=1,
        shuffle=False, num_workers=0, collate_fn=coll)

    ckpt_filename = join(args.result_path, 'ckpt', args.ckpt_name)
    
    def load_ckpt(ckpt_filename):
        return torch.load(ckpt_filename)['state_dict']

    ckpt = load_ckpt(ckpt_filename)

    decoder = Decoder(args, ckpt, data_root)
    save_path = join(args.result_path, f'decode/{args.ckpt_name}')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # decoding
    ext_list = []
    cur_idx = 0
    start = time()
    with torch.no_grad():
        for raw_article_batch in loader:
            tokenized_article_batch = map(tokenize(None), raw_article_batch)
            for raw_art_sents in tokenized_article_batch:
                ext_idx = decoder(raw_art_sents)
                ext_list.append(ext_idx)
                cur_idx += 1
                print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                        cur_idx, n_data, cur_idx/n_data*100, timedelta(seconds=int(time()-start))
                ), end='')
    print()

    # write files
    names, _ = list_data(data_path)
    file_idxs = [name.split('.')[0] for name in names]
    for file_idx, ext_ids in zip(file_idxs,ext_list):
        dec = []
        art_path = join(data_path, f'{file_idx}.json')
        with open(art_path) as f:
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
    ref_path = join(data_root, 'refs/{}'.format(split))
    print("eval_rouge")
    ROUGE = eval_rouge(dec_path, ref_path)
    print(ROUGE)
    with open(join(args.result_path, f'ROUGE/{args.ckpt_name}.txt'), 'w') as f:
        print(ROUGE, file=f)

def get_encoded(args, split):

    # setup loader
    def coll(batch):
        articles = list(filter(bool, batch))
        return articles
    data_root= join(args.project_path,DATA_DIR)
    data_path = join(data_root,split)


    dataset = DecodeDataset(data_path)
    n_data = len(dataset)

    loader = DataLoader(dataset, batch_size=1,
        shuffle=False, num_workers=0, collate_fn=coll)

    ckpt_filename = join(args.result_path, 'ckpt', args.ckpt_name)
    
    def load_ckpt(ckpt_filename):
        return torch.load(ckpt_filename)['state_dict']

    ckpt = load_ckpt(ckpt_filename)

    def del_key(state_dict, key):
        try:
            del state_dict[key]
        except KeyError:
            pass
    for key in ['_ws.weight', '_ws.bias']:
        del_key(ckpt, key)

    encoder = SummarizerEncoder(args.emb_dim, args.vocab_size, args.conv_hidden,
                                args.encoder_hidden, args.encoder_layer, data_root).to('cuda')
    encoder.load_state_dict(ckpt)

    enc_list = []
    cur_idx = 0
    start = time()
    with torch.no_grad():
        for raw_article_batch in loader:
            tokenized_article_batch = map(tokenize(None), raw_article_batch)
            for raw_art_sents in tokenized_article_batch:
                enc_out = encoder(raw_art_sents)
                enc_list.append(enc_out)
                cur_idx += 1

    return enc_list

def get_labels(args, split):
    data_root= join(args.project_path,DATA_DIR)
    data_path = join(data_root,split)

    def coll(batch):
        lab_t_batch = []
        for label, num_sents in batch:
            lab_t = torch.zeros(num_sents,1,device='cuda',dtype=torch.long).to('cuda')
            for sent_idx in label:
                lab_t[sent_idx-1, 0] = 1
            lab_t_batch.append(lab_t)
        return lab_t_batch
    
    label_set = DecodeLabels(data_path)
    loader = DataLoader(label_set, batch_size=1, shuffle=False, num_workers=0, collate_fn=coll)

    labels = [label[0] for label in loader]

    return labels

class argWrapper(object):
  def __init__(self,
               ckpt_name,
               result_path='./result',
               project_path='.',
               batch=1,
               cuda=torch.cuda.is_available(),
               emb_dim=128,
               vocab_size=30004,
               conv_hidden=100,
               encoder_layer=12,
               encoder_hidden=512,
               ):
    self.ckpt_name = ckpt_name
    self.result_path = result_path
    self.project_path = project_path
    self.batch = batch
    self.cuda = cuda
    self.emb_dim = emb_dim
    self.vocab_size = vocab_size
    self.conv_hidden = conv_hidden
    self.encoder_layer = encoder_layer
    self.encoder_hidden = encoder_hidden