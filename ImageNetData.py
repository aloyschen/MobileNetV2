import config
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class DataReader:
    def __init__(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            config.train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        self.train_dataloader = DataLoader(
            train_dataset, batch_size = config.train_batch, shuffle = True, num_workers = 4, pin_memory = True)

        self.val_dataloader = DataLoader(
            datasets.ImageFolder(config.val_dir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size = config.val_batch, shuffle = False,
            num_workers = 4, pin_memory=True)


