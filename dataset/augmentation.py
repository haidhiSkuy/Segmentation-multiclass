import albumentations as A  
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(266,266), 
    A.HorizontalFlip(p=0.5), 
    A.VerticalFlip(p=0.5), 
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5), 
])

val_transform = A.Compose([
    A.Resize(266,266),
])

def get_augmentation() -> dict: 
    augmentation_dict = { 
        'train': train_transform, 
        'val': val_transform
    }
    return augmentation_dict 



