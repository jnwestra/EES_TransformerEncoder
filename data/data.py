""" CNN/DM dataset"""
import json
import re
import os
from os.path import join

from torch.utils.data import Dataset


class CnnDmDataset(Dataset):
    def __init__(self, split: str, path: str) -> None:
        assert split in ['train', 'val', 'test']
        self._data_path = join(path, split)
        self._n_data = _count_data(self._data_path)
        self._prev_js = None

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: int):
        if not os.path.exists(join(self._data_path, '{}.json'.format(i))):
            if self._prev_js is not None:
                return self._prev_js # Return previous item (if exists) if the current doesnt exist
            else:
                return self.__getitem__(i+1)
        with open(join(self._data_path, '{}.json'.format(i))) as f:
            js = json.load(f)
            self._prev_js = js
        return js


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data
