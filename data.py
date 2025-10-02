import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2

class AlbumentationsTransform:
    def __init__(self, train=True):
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2470, 0.2435, 0.2616)

        if train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(0.1, 0.1, 15, p=0.5),
                A.CoarseDropout(
                    max_holes=1, max_height=16, max_width=16,
                    min_holes=1, min_height=16, min_width=16,
                    fill_value=tuple(int(x*255) for x in self.mean),
                    p=0.5
                ),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])

    def __call__(self, img):
        return self.transform(image=np.array(img))['image']

def get_dataloaders(batch_size=128):
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                     transform=AlbumentationsTransform(train=True))
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                    transform=AlbumentationsTransform(train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader
