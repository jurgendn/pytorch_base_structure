from typing import Callable, List, Optional, Union

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class SampleDataset(Dataset):
    def __init__(
        self,
        x: Union[List, torch.Tensor],
        y: Union[List, torch.Tensor],
        transforms: Optional[Callable] = None,
    ) -> None:
        super(SampleDataset, self).__init__()
        self.x = x
        self.y = y

        if transforms is None:
            # Replace None with some default transforms
            # If image, could be an Resize and ToTensor
            self.transforms = lambda x: x
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index: int):
        x = self.x[index]
        y = self.y[index]

        x = self.transforms(x)
        return x, y


class SampleDataModule(LightningDataModule):
    def __init__(
        self,
        x: Union[List, torch.Tensor],
        y: Union[List, torch.Tensor],
        transforms: Optional[Callable] = None,
        val_ratio: float = 0,
        batch_size: int = 32,
    ) -> None:
        super(SampleDataModule, self).__init__()
        assert 0 <= val_ratio < 1
        assert isinstance(batch_size, int)
        self.x = x
        self.y = y

        self.transforms = transforms
        self.val_ratio = val_ratio
        self.batch_size = batch_size

        self.setup()
        self.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def prepare_data(self) -> None:
        n_samples: int = len(self.x)
        train_size: int = n_samples - int(n_samples * self.val_ratio)

        self.train_dataset = SampleDataset(
            x=self.x[:train_size], y=self.y[:train_size], transforms=self.transforms
        )
        if train_size < n_samples:
            self.val_dataset = SampleDataset(
                x=self.x[train_size:], y=self.y[train_size:], transforms=self.transforms
            )
        else:
            self.val_dataset = SampleDataset(
                x=self.x[-self.batch_size :],
                y=self.y[-self.batch_size :],
                transforms=self.transforms,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_dataset, batch_size=self.batch_size)
