import torchvision
import lightning as L
from torch.utils.data import DataLoader
from utils.transforms import train_transform, test_transform


class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="~/data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


class CIFARDataModule(L.LightningDataModule):
    def __init__(
        self, data_dir="data", batch_size=512, shuffle=True, num_workers=4
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None):
        self.train_dataset = Cifar10SearchDataset(
            root=self.data_dir, train=True, transform=train_transform
        )

        self.val_dataset = Cifar10SearchDataset(
            root=self.data_dir, train=False, transform=test_transform
        )

        self.test_dataset = Cifar10SearchDataset(
            root=self.data_dir, train=False, transform=test_transform
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )
