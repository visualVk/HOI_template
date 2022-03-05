import os
from dataset import build_dataset
from dataset.hicodet import nested_tensor_collate
from torch.utils.data import DataLoader
from config import config
if __name__ == '__main__':
    train_image_dir = os.path.join(
        config.DATASET.ROOT,
        config.DATASET.NAME,
        config.DATASET.TRAIN_IMAGES)
    dataset = build_dataset(
        'hico_det',
        train_image_dir,
        'D:\\code\\dpl\\data_an\\hico_train.json')

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=nested_tensor_collate)

    for nested_tensor, targets in dataloader:
        tensors, masks = nested_tensor.decompose()
        print(tensors.shape, masks.shape)
