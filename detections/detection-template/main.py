import pytorch_lightning as pl
from torchvision import transforms

from dataset import Dataset
# from data_module import MyDataModule
# from model import MyModel


if __name__=="__main__":

    transform = transforms.Compose([transforms.ToTensor(),])
    data_root = "/home/steffen/dev/ad-nn-workshop/detections/detection-data"
    
    train_dataset = Dataset(data_root, "train", transform)
    val_dataset = Dataset(data_root, "val", transform)
    
    print(train_dataset[0])
    
    
    # # Initialize datamodule
    # dm = MyDataModule()

    # # Initialize model
    # model = MyModel()

    # # Initialize trainer
    # trainer = pl.Trainer()

    # # Run training
    # trainer.fit(model, dm)
