""" IMGDM dataset"""
import json
import re
import os
from os.path import join, isfile

from torch.utils.data import Dataset

class ImgDmDataset(Dataset):
    def __init__(self, path: str, log_file: _io.TextIOWrapper) -> None:
        self._data_path = path
        self._names, self._n_data = list_data(self._data_path)

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: int):
        with open(join(self._data_path, self._names[i])) as f:
            js = json.loads(f.read())
        log_file.write(f'{_names[i]}\n')
        return js

def list_data(path):
    """ get names of and count number of data in the given path"""
    names = [filename for filename in os.listdir(path) if isfile(filename) and filename.endswith('.json')]
    n_data = len(names)
    return names, n_data