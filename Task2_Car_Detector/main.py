import os
import sys
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader
from utils import extended_collate

from data_module import DataModule
from dataset import ObjectDetectionDataset
from module import ObjectDetector


def parse_args():
    parser = argparse.ArgumentParser("NN-Workshop")
    parser.add_argument("--data_dir", "-d", default=None)
    parser.add_argument("--save_imgs", default=False, action="store_true")

    args = parser.parse_args(sys.argv[1:])
    if len(sys.argv) < 1:
        parser.print_help()

    return args


if __name__=="__main__":

    # Get arguments
    args = parse_args()

    # Check local data directory exits
    if not os.path.exists(args.data_dir):
        raise Exception("Data directory does not exist")

    # data_root = "/home/steffen/data/nn-workshop-data"

    # Define logs & checkpoints
    logger = [TensorBoardLogger("~/pytorch_logs", name="Workshop")]
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="checkpoint-{epoch:02d}-{val_total_loss:.3f}",
        save_top_k=10,
        mode="min",
    )

    
    ####### TODO: Define parameters #########
    batch_size = 64
    num_threads = 8
    #########################################


    ###### TODO: Initialize dataset #########
    data_train = None
    data_val = None
    #########################################


    #### TODO: Initialize dataloader ########
    train_dataloader = None
    val_dataloader = None
    #########################################


    # Alternative option initialize DataModule


    # Initialize model


    # Initialize trainer


    # Run training
