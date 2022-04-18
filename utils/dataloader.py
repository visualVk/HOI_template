import os
from typing import Optional, Callable
import os.path as osp
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, DistributedSampler, ConcatDataset

from dataset.hico_det import HICODetDataset
from utils.logger import log_every_n
from transforms import TrainTransform, EvalTransform
from utils.logger import log_every_n


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


def get_dataset(cfg):
    train_transform = TrainTransform(
        mean=cfg.DATASET.MEAN,
        std=cfg.DATASET.STD,
        scales=cfg.DATASET.SCALES,
        max_size=cfg.DATASET.MAX_SIZE
    )
    eval_transform = EvalTransform(
        mean=cfg.DATASET.MEAN,
        std=cfg.DATASET.STD,
        max_size=cfg.DATASET.MAX_SIZE
    )

    data_root = cfg.DATASET.ROOT  # abs path in yaml
    # get train data list
    train_root = osp.join(data_root, 'train2015')
    train_set = [
        d for d in os.listdir(train_root) if osp.isdir(
            osp.join(
                train_root,
                d))]
    if len(train_set) == 0:
        train_set = ['.']
    train_list = []
    for sub_set in train_set:
        train_sub_root = osp.join(train_root, sub_set)
        log_every_n(20, f'==> load train sub set: {train_sub_root}')
        train_sub_set = HICODetDataset(cfg, train_sub_root, train_transform)
        train_list.append(train_sub_set)
    # get eval data list
    eval_root = osp.join(data_root, 'test2015')
    eval_set = [
        d for d in os.listdir(eval_root) if osp.isdir(
            osp.join(
                eval_root,
                d))]
    if len(eval_set) == 0:
        eval_set = ['.']
    eval_list = []
    for sub_set in eval_set:
        eval_sub_root = osp.join(eval_root, sub_set)
        log_every_n(20, f'==> load val sub set: {eval_sub_root}')
        eval_sub_set = HICODetDataset(cfg, eval_sub_root, eval_transform)
        eval_list.append(eval_sub_set)
    # concat dataset list
    train_dataset = list_to_set(train_list, 'train')
    eval_dataset = list_to_set(eval_list, 'eval')

    return train_dataset, eval_dataset


def list_to_set(data_list, name='train'):
    if len(data_list) == 0:
        dataset = None
        log_every_n(40, f"{name} dataset is None")
    elif len(data_list) == 1:
        dataset = data_list[0]
    else:
        dataset = ConcatDataset(data_list)

    if dataset is not None:
        log_every_n(20, f'==> the size of {name} dataset is {len(dataset)}')
    return dataset
