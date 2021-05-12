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
from decoding import Decoder, DecodeDataset
from evaluate import eval_rouge

from data.data import ImgDmDataset
from data.batcher import tokenize

import warnings
warnings.filterwarnings("ignore", category=Warning)

import sys
sys.setrecursionlimit(10000)

DATA_DIR = 'IMGDM'

def test(args, split):

    #setup loader
    def coll(batch):
        articles = list(filter(bool, batch))
        return articles

    data_path = join(args.project_path,DATA_DIR,split)
    
    dataset = DecodeDataset(data_path)

    n_data = len(dataset)
    loader = DataLoader(dataset, batch_size=args.batch,
        shuffle=False, num_workers=2 if args.cuda else 0, collate_fn=coll)

    print('dataset length', n_data)

    # decode and eval top 5 models
    if not os.path.exists(join(args.result_path, 'decode')):
        os.mkdir(join(args.result_path, 'decode'))
        
    if not os.path.exists(join(args.result_path, 'ROUGE')):
        os.mkdir(join(args.result_path, 'ROUGE'))
    
    ckpt_filename = join(args.result_path, 'ckpt', args.ckpt_name)
    
    def load_ckpt(ckpt_filename):
        return torch.load(ckpt_filename)['state_dict']

    ckpt = load_ckpt(ckpt_filename)

    decoder = Decoder(args, ckpt)
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
    for file_idx, ext_ids in enumerate(ext_list):
        dec = []
        data_path = join(project_path, DATA_DIR, '{}/{}.json'.format(split, file_idx))
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
    ref_path = join(args.project_path, DATA_DIR, 'refs/{}'.format(split))
    print("eval_rouge")
    ROUGE = eval_rouge(dec_path, ref_path)
    print(ROUGE)
    with open(join(args.result_path, f'ROUGE/{args.ckpt_name}.txt'), 'w') as f:
        print(ROUGE, file=f)

def get_encoded(args, split):

    if not exists(args.log_path):
        os.makedirs(args.log_path)

    log_filename = join(args.log_path, f'summ_{str(time())}.txt')

    with open(log_filename,'w') as log_file:

        # setup loader
        def coll(batch):
            articles = list(filter(bool, batch))
            return articles

        data_path = join(args.project_path,DATA_DIR,split)

        log_file.write(f'Data stored in {data_path}\n')

        dataset = DecodeDataset(data_path)
        n_data = len(dataset)

        log_file.write(f'Dataset length: {n_data}\n')

        loader = DataLoader(dataset, batch_size=args.batch,
            shuffle=False, num_workers=2 if args.cuda else 0, collate_fn=coll)

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
                                    args.encoder_hidden, args.encoder_layer)
        encoder.load_state_dict(ckpt)

        enc_list = []
        cur_idx = 0
        start = time()
        log_file.write(f'Getting encoded article sentences at {str(timedelta(seconds=start%86400))}\n')
        with torch.no_grad():
            for raw_article_batch in loader:
                tokenized_article_batch = map(tokenize(None,raw_article_batch), raw_article_batch)
                for raw_art_sents in tokenized_article_batch:
                    enc_out, name = encoder(raw_art_sents)
                    log_file.write(f'{name}\n')
                    enc_list.append(enc_out)
                    cur_idx += 1
                    log_file.write('{}/{} ({:.2f}%) encoded in {} seconds\r'.format(
                            cur_idx, n_data, cur_idx/n_data*100, timedelta(seconds=int(time()-start))
                    ), end='')

        log_file.write('-'*20 + '\n')
        log_file.close()

    return enc_list

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
               log_path='./log.txt'):
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
    self.log_path = log_path