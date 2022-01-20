import torch
import torch.utils.data as data

import numpy as np


class Dataset(data.Dataset):
    """ Dataset wrapper class """

    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data = np.loadtxt(data_dir)

    def __getitem__(self, index):
        """ Get item based on index """
        data_point = self.data[index]
        return data_point

    def __len__(self):
        return self.total_size
