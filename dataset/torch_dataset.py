import cv2  
import torch 
import numpy as np
from torch.utils.data import Dataset 

def map_mask(msk: np.ndarray): 
    grouped_classes = {
        (155, 38, 182): 0,
        (14, 135, 204): 1,
        (124, 252, 0): 2,
        (255, 20, 147): 3,
        (169, 169, 169): 4,
    } 
    
    mapped_image = np.zeros(msk.shape[0:2], dtype=np.int64)

    # Iterate over each pixel and map it to the corresponding class
    for i in range(msk.shape[0]):
        for j in range(msk.shape[1]):
            pixel_value = tuple(msk[i, j])
            if pixel_value in grouped_classes:
                mapped_image[i, j] = grouped_classes[pixel_value]
    
    return mapped_image


class DroneDataset(Dataset): 
    def __init__(self, image_list, mask_list, transform):  
        self.image_list = image_list 
        self.mask_list = mask_list 
        self.transform = transform 
    
    def __len__(self): 
        return len(self.image_list) 
    
    def __getitem__(self, index):
        img_path = self.image_list[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        msk_path = self.mask_list[index] 
        msk = cv2.imread(msk_path, cv2.IMREAD_UNCHANGED)
        msk = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB) 

        if self.transform: 
            transformed = self.transform(image=img, mask=msk)
            image = transformed["image"] / 255.0
            mask = transformed["mask"]  

        mask = map_mask(mask)
        
        return image, mask 
    