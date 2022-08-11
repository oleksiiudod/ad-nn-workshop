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


if __name__ == "__main__":

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

    data_train = ObjectDetectionDataset(args.data_dir, "train")
    data_val = ObjectDetectionDataset(args.data_dir, "val")
    #########################################

    #### TODO: Initialize dataloader ########
    train_dataloader = None
    val_dataloader = None

    train_dataloader = DataLoader(
        data_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_threads,
        collate_fn=extended_collate,
    )

    val_dataloader = DataLoader(
        data_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_threads,
        collate_fn=extended_collate,
    )

    #########################################

    # Alternative option initialize DataModule
    dm = DataModule(batch_size=32, data_dir=args.data_dir)

    # Initialize model
    model = ObjectDetector(save_imgs=args.save_imgs)

    # Initialize trainer
    trainer = pl.Trainer(logger=logger, callbacks=[checkpoint_callback])

    # Run training
    trainer.fit(model, train_dataloader, val_dataloader)
    # trainer.fit(model, dm)
