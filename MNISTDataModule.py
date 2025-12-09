import torch
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


class MNISTDataModule(pl.LightningDataModule):
    """
    A clean PyTorch Lightning DataModule for MNIST.

    Responsibilities:
    - Download the MNIST dataset
    - Compute dataset statistics (mean, std, max) from the raw training set
    - Re-create datasets with proper normalization
    - Provide train/val/test DataLoaders
    """

    def __init__(self, batch_size: int = 128, data_dir: str = "./MNIST"):
        super().__init__()

        self.data_dir = ''
        self.batch_size = batch_size

        # Dataset statistics (computed in setup)
        self.mean = 0.5
        self.std = 1.0
        self.max_val = 1.0

        # Datasets (populated inside setup_data)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """
        Download MNIST once per node.
        Lightning ensures this is called only on the main process.
        """
        # Download MNIST if not already
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup_data(self):
        """
        Set up datasets.

        Steps:
        1. Load dataset with only ToTensor() so we can compute raw stats.
        2. Compute mean/std/max over the training set.
        3. Re-load datasets using the correct normalization transform.
        """

        # 1.Initial transform just to load tensors
        initial_transform = transforms.ToTensor()
        full_train = datasets.MNIST(self.data_dir, train=True, transform=initial_transform)

        # 2.Compute stats on raw (un-normalized) train data
        mean, std, max_val = self.compute_stats(DataLoader(full_train, batch_size=self.batch_size))
        self.mean = mean
        self.std = std
        self.max_val = max_val

        # 3. Define the actual transform with normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self.mean,), (self.std,))
        ])

        # Reload datasets with the final transform
        full_train = datasets.MNIST(self.data_dir, train=True, transform=transform)
        self.train_dataset, self.val_dataset = random_split(full_train, [55000, 5000])
        self.test_dataset = datasets.MNIST(self.data_dir, train=False, transform=transform)

    def train_dataloader(self):
        """Return the training DataLoader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """Return the validation DataLoader."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """Return the test DataLoader."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    @staticmethod
    def compute_stats( dataloader):

        """
        Compute mean, std, and max pixel value of the training set.

        Args:
            dataloader: DataLoader containing the raw (un-normalized) training data.

        Returns:
            mean (float): Average pixel value across entire training dataset.
            std (float): Standard deviation.
            max_val (float): Maximum raw pixel value (used to check scaling).
        """

        sum_ = 0.0
        sum_sq = 0.0
        n = 0
        global_max = float('-inf')

        # Loop through batches and accumulate statistics
        for x, _ in dataloader:
            x = x.float()
            n += x.size(0)
            sum_ += x.sum(dim=[0, 2, 3])
            sum_sq += (x ** 2).sum(dim=[0, 2, 3])
            global_max = max(global_max, x.max().item())

        # Number of total pixels: N * H * W
        total_pixels = n * x.size(2) * x.size(3)
        # Compute final mean and std
        mean = sum_ / total_pixels
        std = torch.sqrt(sum_sq / total_pixels - mean ** 2)

        return mean.item(), std.item(), global_max
