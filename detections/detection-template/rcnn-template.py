import os

import torch
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchmetrics import Accuracy


class FasterRcnnFpn(LightningModule):
    def __init__(self):
        super().__init__()

        model_params = {
            "pretrained": False,
            "progress": True,
            "num_classes": self.num_detected_classes,
            "pretrained_backbone": True,
            "trainable_backbone_layers": 5,
        }
        self.model = fasterrcnn_resnet50_fpn(**model_params)

        self.accuracy = Accuracy()

    def forward(self, x):
        self.model.eval()
        results = self.model(x)
        self.model.train()
        return results

    def training_step(self, batch, batch_idx):
        x, y = self.process_batch(batch)

        loss_dict = self.model(x, y)
        loss = sum(loss for loss in loss_dict.values())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = self.process_batch(batch)
        preds = self.forward(x)
        self.accuracy(preds, y)

        # calculating validation loss is not possible for the current torchvision.models
        # implementation without impacting training process
        self.log("val_acc", self.accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def process_batch(self, batch):
        images, bboxes, labels = batch[0]

        image_list = []
        annotations_list = []

        # Loop through batch
        for i in range(len(images)):

            # TODO
            # Add here some clean-up and data preprocessing and possibly augmentations

            image_list.append(torch.tensor(images[i], device=self.device))

            annotations_list.append(
                {
                    "boxes": torch.tensor(bboxes[i], device=self.device),
                    "labels": torch.tensor(labels[i], device=self.device),
                }
            )

        image_list = torch.stack(image_list)

        return image_list, annotations_list
