import torchdata
from s3torchconnector import S3IterableDataset
from torch.utils.data.datapipes.datapipe import IterDataPipe


def get_s3iterabledataset(s3_uri: str, region: str, load_sample) -> IterDataPipe:
    dataset = S3IterableDataset.from_prefix(s3_uri, region=region)
    dataset = torchdata.datapipes.iter.IterableWrapper(dataset)

    return dataset.map(load_sample)
