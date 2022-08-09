import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torchvision import transforms

from utils import extended_collate
from dataset import ObjectDetectionDataset


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, transform, data_dir: str = "./"):
        super().__init__()

        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_threads = 8

        # Define any augmentation/preprocessing transforms
        self.transform = transform
        
        ###### TODO: Define your dataset ########
        self.data_train = None
        self.data_val = None
        #########################################

        self.data_train = ObjectDetectionDataset(self.data_dir, "train", self.transform)
        self.data_val = ObjectDetectionDataset(self.data_dir, "val", self.transform)

    def prepare_data(self):
        """
        Any preparatory steps that need to be done regarding the 
        data. Will be done once before training in a single process. 
        Could include things like downloading data etc.
        """
        # Note that if it doesn't apply to your use case then just skip
        pass

    def setup(self, stage = None):
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
    def train_dataloader(self):
        """
        Create the training dataloader that will parse the dataset.
        Usually it just acts to wrap the dataset class defined previously.
        """
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_threads,
            collate_fn=extended_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_threads,
            collate_fn=extended_collate,
        )