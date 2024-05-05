import os
import lightning as L
from .torch_dataset import DroneDataset
from torch.utils.data import DataLoader
from augmentation import get_augmentation
from sklearn.model_selection import train_test_split


class DroneDataModule(L.LightningDataModule):
    def __init__(self, image_dir: float, mask_dir: float,train_size: float = 0.8,batch_size: int = 32):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.train_size = train_size
        self.batch_size = batch_size

        self.transform = get_augmentation()

    def setup(self, stage: str):
        images = sorted([os.path.join(self.image_dir, i) for i in os.listdir(self.image_dir)])
        masks = sorted([os.path.join(self.mask_dir, i) for i in os.listdir(self.mask_dir)]) 

        x_train, x_test, y_train, y_test = train_test_split(images, masks, train_size=self.train_size) 
        self.train_dataset = DroneDataset(x_train, y_train, self.transform['train'])
        self.val_dataset = DroneDataset(x_test, y_test, self.transform['val'])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size) 