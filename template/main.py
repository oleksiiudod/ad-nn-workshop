import pytorch_lightning as pl

from data_module import MyDataModule
from model import MyModel


if __name__=="__main__":
    
    # Initialize datamodule
    dm = MyDataModule()

    # Initialize model
    model = MyModel()

    # Initialize trainer
    trainer = pl.Trainer()

    # Run training
    trainer.fit(model, dm)
