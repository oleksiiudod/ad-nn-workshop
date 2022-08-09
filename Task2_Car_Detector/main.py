import os
import sys
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torchvision import transforms

from data_module import DataModule
from model import MyModel


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

    # Define Augmentation 
    transform = transforms.Compose([transforms.ToTensor(),])

    # Define logs & checkpoints
    logger = [TensorBoardLogger("~/pytorch_logs", name="Workshop")]
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="checkpoint-{epoch:02d}-{val_total_loss:.3f}",
        save_top_k=10,
        mode="min",
    )

    # Initialize DataModule
    dm = DataModule(batch_size=32, transform=transform, data_dir=args.data_dir)

    # Initialize model
    model = MyModel(save_imgs=args.save_imgs)

    # Initialize trainer
    trainer = pl.Trainer(logger=logger, callbacks=[checkpoint_callback])

    # Run training
    trainer.fit(model, dm)
