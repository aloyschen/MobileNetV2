import config
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import calc_dataset_stats
class CIFAR100data:
    def __init__(self):
        mean, std = calc_dataset_stats(torchvision.datasets.CIFAR100(root = './data', train = True, download = True).train_data, axis = (0, 1, 2))
        train_transform = transforms.Compose(
            [transforms.RandomCrop(config.image_height),
             transforms.RandomHorizontalFlip(),
             transforms.ColorJitter(0.3, 0.3, 0.3),
             transforms.ToTensor(),
             transforms.Normalize(mean = mean, std = std)]
        )

        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean = mean, std = std)]
        )

        self.train_dataloader = DataLoader(torchvision.datasets.CIFAR100(root = './data', train = True, download = True, transform = train_transform), batch_size = config.train_batch, shuffle = True, num_workers = 10)
        self.val_dataloader = DataLoader(torchvision.datasets.CIFAR100(root = './data', train = False, download = True, transform = test_transform), batch_size = config.val_batch, shuffle = False, num_workers = 10)


