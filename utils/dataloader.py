from typing import Optional, Callable

from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, DistributedSampler


def split_train_dataset(dataset: Dataset, p: int):
    if p == 1:
        return dataset

    n = len(dataset)
    train_len = int(n * p)
    eval_len = n - train_len
    train_dataset, eval_dataset = random_split(dataset, [train_len, eval_len])
    return train_dataset, eval_dataset


def dataset_to_dataloader(
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        ddp: bool, collate_fn: Optional[Callable] = None):
    if not ddp:
        sampler = RandomSampler(dataset)
    else:
        sampler = DistributedSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size,
        False,
        sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True)
    return dataloader


def build_dataloader(
        train_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        collate_fn: Optional[Callable] = None,
        ddp: bool = True,
        num_workers: int = 16,
        batch_size: int = 16,
        p: int = 0.6):
    train_dataset, val_dataset = split_train_dataset(train_dataset, p)
    dataset_list = [train_dataset, val_dataset, test_dataset]

    result = []
    for dataset in dataset_list:
        if dataset is None:
            continue
        result.append(
            dataset_to_dataloader(
                dataset,
                batch_size,
                num_workers,
                ddp, collate_fn))

    return result
