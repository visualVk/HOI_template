import json
import os.path
from typing import Optional, Callable
from torchvision import transforms as tfs
import transforms as wtfs

import numpy as np

from constants import BBOX
from utils.misc import nested_tensor_from_tensor_list
from utils.box_ops import *

from dataset.base import ImageDataset

human_id = 0


class HICODet(ImageDataset):
    def __init__(
            self,
            root: str,
            anno_file: str,
            bbox_type: str = BBOX.XYXY,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None) -> None:
        super(
            HICODet,
            self).__init__(
            root,
            transform,
            target_transform,
            transforms)

        with open(anno_file) as fp:
            anno = json.load(fp)

        self.num_object_cls = 81
        self.num_interaction_cls = 600
        self.num_action_cls = 117

        self._load_annotation_and_metadata(anno)

    def __len__(self):
        # return len(self._idx)
        return 10

    def __getitem__(self, i):
        intra_idx = self._idx[i]
        filename = self._filename(intra_idx)
        target = self._hoi_set(intra_idx)
        image = self.load_image(os.path.join(self._root, filename))
        return self._transforms(image, target)

    def _filename(self, i):
        return self._filenames[i]

    def _hoi_set(self, i: int) -> dict:
        hoi_set = self._anno[i]
        interaction = hoi_set['hoi']
        labels_v = hoi_set['verb']
        boxes_h = hoi_set['boxes_h']
        boxes_o = hoi_set['boxes_o']
        labels_h = [human_id] * len(interaction)
        labels_o = hoi_set['object']

        # for box_h in hoi_set['boxes_h']:
        #     bbox_human = self._transform_coordinate(box_h)
        #     boxes_h.append(bbox_human)
        # for box_o in hoi_set['boxes_o']:
        #     bbox_object = self._transform_coordinate(box_o)
        #     boxes_o.append(bbox_object)

        boxes_h = torch.from_numpy(np.array(boxes_h).astype(np.float32))
        boxes_o = torch.from_numpy(np.array(boxes_o).astype(np.float32))
        interaction = torch.from_numpy(
            np.array(interaction).astype(np.float32))
        labels_h = torch.from_numpy(np.array(labels_h).astype(np.float32))
        labels_o = torch.from_numpy(np.array(labels_o).astype(np.float32))
        labels_v = torch.from_numpy(np.array(labels_v).astype(np.float32))

        hoi_set = dict(
            boxes_h=boxes_h,
            boxes_o=boxes_o,
            interaction=interaction,
            labels_v=labels_v,
            labels_h=labels_h,
            labels_o=labels_o)
        return hoi_set

    # def _transform_coordinate(self, coordinate: list):
    #     if not (coordinate[0] <= coordinate[2]
    #             and coordinate[1] <= coordinate[3]):
    #         raise ValueError(
    #             "coordinate([x1, y1, x2, y2]) must [x1, y1] <= [x2, y2]")
    #     if self.bbox_type == BBOX.XYXY:
    #         return coordinate
    #     elif self.bbox_type == BBOX.CXCYHW:
    #         w, h = coordinate[2] - coordinate[0],\
    #             coordinate[3] - coordinate[1]
    #         cx, cy = (coordinate[2] + coordinate[0]) * 0.5,\
    #                  (coordinate[3] + coordinate[1]) * 0.5
    #         return [cx, cy, h, w]
    #     elif self.bbox_type == BBOX.XYHW:
    #         w, h = coordinate[2] - coordinate[0],\
    #             coordinate[3] - coordinate[1]
    #         x, y = coordinate[:2]
    #         return [x, y, h, w]

    def _load_annotation_and_metadata(self, f: dict,):
        anno = f['annotation']
        idx = list(range(len(anno)))

        num_anno = [0 for _ in range(self.num_interaction_cls)]
        empty_ids = f['empty_ids']
        for empty_id in empty_ids:
            idx.remove(empty_id)

        for rid, row in enumerate(anno):
            for hoi in row['hoi']:
                interaction_id = hoi
                num_anno[interaction_id] += 1

        self._anno = anno
        self._idx = idx
        self._num_anno = num_anno
        self._objects = f['objects']
        self._verbs = f['verbs']
        self._filenames = f['filenames']


def custom_collate(batch):
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return images, targets


def nested_tensor_collate(batch):
    images, targets = [], []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return images, targets


def make_hico_transforms(image_set='train'):
    normalize = wtfs.Compose([
        wtfs.ToTensor(input_format='comb'),
        wtfs.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return wtfs.Compose([
            wtfs.RandomHorizontalFlip(),
            wtfs.RandomSelect(
                wtfs.RandomResize(scales, max_size=1333),
                wtfs.Compose([
                    wtfs.RandomResize([400, 500, 600]),
                    wtfs.RandomSizeCrop(384, 600),
                    wtfs.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return wtfs.Compose([
            wtfs.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_hico_det(root: str, anno_file: str):
    transforms = make_hico_transforms()
    # hico_det = HICODet(
    #     root,
    #     anno_file,
    #     transform=wtfs.ToTensor('pil'),
    #     target_transform=wtfs.ToTensor(
    #         input_format='dict'))
    hico_det = HICODet(root, anno_file, transforms=transforms)
    return hico_det
