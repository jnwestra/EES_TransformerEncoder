""" IMGDM dataset"""
import json
import re
import os
from os.path import join, isfile

from torch.utils.data import Dataset

def get_names(path):
    """ get names of and count number of data in the given path"""
    names = [filename for filename in os.listdir(path) if isfile(filename) and filename.endswith('.json')]
    n_data = len(names)
    return names, n_data

class ImgDmDataset(Dataset):
    def __init__(self, split: str, path: str) -> None:
        assert split in ['train', 'val', 'test']
        self._data_path = join(path, split)
        self._names, self._n_data = get_names(self._data_path)

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: int):
        with open(join(self._data_path, self._names[i])) as f:
            js = json.loads(f.read())
        return js