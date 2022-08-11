import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pytorch_lightning as pl
from backbones.resnet import ResNet, ResidualBlock

from utils import save_example


class ObjectDetector(pl.LightningModule):
    def __init__(self, save_imgs=False):
        super().__init__()

        # Data parameters
        self.input_size = [1280, 128]
        self.output_size = [160, 16]

        #### TODO: Define your model ########

        # Parameters
        self.learning_rate = None

        # Feature extractor

        # Detection head (optional)

        #####################################

        # Code related to saving images (Do not change)
        self.epoch_count = 0
        self.save_imgs = save_imgs
        if self.save_imgs:
            path_base = "./logged_images/"
            print(path_base)
            os.makedirs(path_base, exist_ok=True)

            # Create run folder
            run_id = 0
            while True:
                self.run_path = os.path.join(path_base, "run-{}".format(run_id))
                if not os.path.exists(self.run_path):
                    os.makedirs(self.run_path, exist_ok=True)
                    break
                else:
                    run_id += 1

    def forward(self, x):
        """
        Defines the forward pass aka the predictions/inferences made by
        the model. The input x is usually the batch of data.
        """
        #### TODO: Create forward pass ######
        output = None

        #####################################

        return output

    def training_step(self, batch, batch_idx):
        """
        This defines the training loop and the operations applied to
        every batch of training data loaded.
        """
        img, annotation = self.process_batch(batch)

        #### TODO: Define training step #####

        # Get model prediction
        prediction = None

        # Calculate loss
        loss = None

        #####################################

        # Log loss
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        This defines the validation loop and the operations applied to
        every batch of validation data loaded.
        """
        img, annotation = self.process_batch(batch)

        #### TODO: Define validation step ####

        # Get model prediction
        prediction = None

        # Calculate loss
        loss = None

        # Optional: Metric (Idea: How many cars captured)
        metric = None

        #####################################

        # Log loss
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        # Save example images
        if self.save_imgs and batch_idx < 3:
            out_path = os.path.join(
                self.run_path,
                "val-image-epoch-{}-image{}.jpg".format(self.epoch_count, batch_idx),
            )
            save_example(img[0], annotation[0], prediction[0], out_path)

        return loss

    def validation_epoch_end(self, outputs):
        self.epoch_count += 1

    def configure_optimizers(self):
        """
        Optimizers regulate how weights and biases are updated in the
        network.
        """
        #### TODO: Define optimizer ########
        # Hint: Adam is always a good choice
        optimizer = None

        ####################################

        return optimizer

    def process_batch(self, batch):
        img, annotation = batch

        #### OPTIONAL: Process batch ######

        ###################################

        return img, annotation
