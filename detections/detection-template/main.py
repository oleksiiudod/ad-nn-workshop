import pytorch_lightning as pl
from torchvision import transforms

from data_module import MyDataModule
from model import MyModel

# from model import MyModel


if __name__ == "__main__":

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    data_root = "/home/steffen/dev/ad-nn-workshop/detections/detection-data"

    # Initialize DataModule
    dm = MyDataModule(batch_size=64, transform=transform, data_dir=data_root)

    # Initialize model
    model = MyModel()

    # # Initialize trainer
    # trainer = pl.Trainer()

    # # Run training
    # trainer.fit(model, dm)
