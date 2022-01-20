import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

import pytorch_lightning as pl

from torchvision import transforms


class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, data_dir: str = "./"):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir

        # Define any augmentation/preprocessing transforms
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        
        ###### Define your dataset HERE ########
        self.data_train = None
        self.data_val = None
        self.data_test = None
        ########################################

    def prepare_data(self):
        """
        Any preparatory steps that need to be done regarding the 
        data. Will be done once before training in a single process. 
        Could include things like downloading data etc.
        """
        # Note that if it doesn't apply to your use case then just skip
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Data operations you might want to perform on every GPU. 
        """
        # Note that if it doesn't apply to your use case then just skip
        pass

    def train_dataloader(self):
        """
        Create the training dataloader that will parse the dataset. 
        Usually it just acts to wrap the dataset class defined previously.
        """
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)