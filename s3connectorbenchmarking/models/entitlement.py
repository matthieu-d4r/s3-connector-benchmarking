from typing import override

from s3torchconnector import S3Reader

from s3connectorbenchmarking.models.model import BenchmarkModel


class Entitlement(BenchmarkModel):
    """
    Entitlement does not train anything. Instead, this model simply reads the binary
    object data from S3, so that we may identify the max achievable throughput for a
    given dataset.
    """

    def __init__(self):
        super().__init__()

    def load_sample(self, sample: S3Reader) -> (str, S3Reader):
        key, data = super().load_sample(sample)
        buffer = data.read()
        return len(buffer), key

    @override
    def train_batch(self, i: int, data, target):
        pass
