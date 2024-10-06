from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from s3torchconnector import S3Reader
from torch.utils.data import DataLoader


@dataclass
class BenchmarkModel(ABC):
    """Abstract class for benchmarking models."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_sample(self, sample: S3Reader) -> (str, S3Reader):
        """Transform a given sample (a file-like Object) to a model's input."""
        return sample.key, sample

    def train(self, epochs: int, dataloader: DataLoader) -> None:
        """Train a model."""
        for _ in range(0, epochs):
            for i, (data, target) in enumerate(dataloader):
                self.train_batch(i, data, target)

    @abstractmethod
    def train_batch(self, i: int, data, target) -> None:
        raise NotImplementedError
