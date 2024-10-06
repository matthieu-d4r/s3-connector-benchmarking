import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils import benchmark
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.datapipes.datapipe import IterDataPipe

from s3connectorbenchmarking.datasets.s3iterabledataset import get_s3iterabledataset
from s3connectorbenchmarking.models import entitlement
from s3connectorbenchmarking.models.model import BenchmarkModel

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG
)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_benchmark(cfg: DictConfig) -> None:
    logging.info("Running benchmark...")
    logging.info(f"Loaded config:\n{OmegaConf.to_yaml(cfg)}")

    model = prepare_model(cfg)
    dataset = prepare_dataset(cfg, model.load_sample)
    dataloader = prepare_dataloader(dataset, cfg.batch_size, cfg.num_workers)

    execute(cfg, model, dataloader)


# FIXME: use enums?
def prepare_dataset(cfg: DictConfig, load_sample) -> IterDataPipe:
    match cfg.dataset_type:
        case "S3IterableDataset":
            return get_s3iterabledataset(cfg.s3_uri, cfg.region, load_sample)
        case _:
            raise ValueError(f"Unsupported dataset type: {cfg.dataset_type}")


# FIXME: use enums?
def prepare_model(cfg: DictConfig) -> BenchmarkModel:
    match cfg.model:
        case "ent":
            return entitlement.Entitlement()
        case _:
            raise ValueError(f"Unsupported model: {cfg.model}")


def prepare_dataloader(
    dataset: Dataset, batch_size: int, num_workers: int
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        # shuffle=False,
        num_workers=num_workers,
        # collate_fn=default_collate,
        # drop_last=False,
    )


def execute(cfg: DictConfig, model: BenchmarkModel, dataloader: DataLoader) -> None:
    label = model.__class__.__name__
    sub_label = f"[{cfg.epochs}]"

    results = []
    for num_threads in [1, 4]:
        results.append(
            benchmark.Timer(
                stmt="model.train(epochs, dataloader)",
                globals={"model": model, "epochs": 1, "dataloader": dataloader},
                label=label,
                sub_label=sub_label,
                description=cfg.model,
                num_threads=num_threads,
            ).blocked_autorange(min_run_time=1)
        )

    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":
    run_benchmark()
