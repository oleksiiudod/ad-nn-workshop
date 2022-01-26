import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pytorch_lightning as pl
from resnet import ResNet, ResidualBlock


class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Optimizer parameters
        self.learning_rate = 1e-3

        ###### Define your model HERE ########
        self.model = ResNet(ResidualBlock, num_classes=1)
        ######################################

    def forward(self, x):
        """
        Defines the forward pass aka the predictions/inferences made by
        the model. The input x is usually the batch of data.
        """
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        """
        This defines the training loop and the operations applied to
        every batch of training data loaded.
        """
        # Image, number of cars

        img, annotation = self.process_batch(batch)
        prediction = self.forward(img)

        annotation = (
            torch.tensor(annotation, dtype=torch.float).unsqueeze(-1).to(self.device)
        )

        print("predictions: ", prediction[:10])
        print("annotation: ", annotation[:10])

        loss = F.mse_loss(annotation, prediction)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

        ############# Example ##############
        # x, y = batch
        # y_prediction = self.forward(x)
        # loss = F.cross_entropy(y_prediction, y)
        # self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # return loss

    def validation_step(self, batch, batch_idx):
        """
        This defines the validation loop and the operations applied to
        every batch of validation data loaded.
        """
        img, annotation = self.process_batch(batch)
        prediction = self.forward(img)

        annotation = (
            torch.tensor(annotation, dtype=torch.float).unsqueeze(-1).to(self.device)
        )

        print("validation predictions: ", prediction[:10])
        print("validation annotation: ", annotation[:10])

        loss = F.mse_loss(annotation, prediction)
        self.log("val loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss
        ############# Example ##############
        # x, y = batch
        # y_prediction = self.forward(x)
        # loss = F.cross_entropy(y_prediction, y)
        # self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # return loss

    def configure_optimizers(self):
        """
        Optimizers regulate how weights and biases are updated in the
        network.
        """
        # Adam (common choice)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def process_batch(self, batch):
        img, annotation = batch

        labels_batch = annotation["labels"]
        new_labels_batch = []

        for i in range(len(labels_batch)):
            labels = labels_batch[i]
            no_cars = 0

            for j in range(len(labels)):
                if labels[j] == 0:
                    no_cars += 1
            new_labels_batch.append(no_cars)

        return img, np.array(new_labels_batch)
        # detection_labelmap = (
        #     "car",
        #     "bus",
        #     "bike",
        #     "motorbike",
        #     "mobility_device",
        #     "people",
        #     "wheel",
        #     "robot",
        #     "stroller",
        #     "scooter_with_person",
        #     "scooter_without_person",
        # )
