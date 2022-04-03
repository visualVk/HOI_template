import torch
import random
import PIL
import torchvision
import torchvision.transforms.functional as F
from utils.box_ops import box_xyxy_to_cxcywh
from torchvision import transforms as tfs

__all__ = [
    'to_tensor',
    'hflip',
    'crop',
    'resize',
    'ColorJitter',
    'RandomHorizontalFlip',
    'RandomAdjustImage',
    'RandomSelect',
    'RandomResize',
    'RandomSizeCrop',
    'ToTensor',
    'Normalize',
    'Compose']


def _to_list_of_tensor(x, dtype=None, device=None):
    return [torch.as_tensor(item, dtype=dtype, device=device) for item in x]


def _to_tuple_of_tensor(x, dtype=None, device=None):
    return tuple(torch.as_tensor(item, dtype=dtype, device=device)
                 for item in x)


def _to_dict_of_tensor(x, dtype=None, device=None):
    return dict([(k, torch.as_tensor(v, dtype=dtype, device=device))
                for k, v in x.items()])


def _to_comb_of_tensor(x, dtype=None, device=None):
    image, target = x
    image = torchvision.transforms.functional.to_tensor(
        image).to(dtype=dtype, device=device)
    # target = _to_dict_of_tensor(target, dtype=dtype, device=device)
    return image, target


def to_tensor(x, input_format='tensor', dtype=None, device=None):
    """Convert input data to tensor based on its format"""
    if input_format == 'tensor':
        if isinstance(x, PIL.Image.Image):
            return F.to_tensor(x)
        return torch.as_tensor(x, dtype=dtype, device=device)
    elif input_format == 'pil':
        return torchvision.transforms.functional.to_tensor(x).to(
            dtype=dtype, device=device)
    elif input_format == 'list':
        return _to_list_of_tensor(x, dtype=dtype, device=device)
    elif input_format == 'tuple':
        return _to_tuple_of_tensor(x, dtype=dtype, device=device)
    elif input_format == 'dict':
        return _to_dict_of_tensor(x, dtype=dtype, device=device)
    elif input_format == 'comb':
        return _to_comb_of_tensor(x, dtype=dtype, device=device)
    else:
        raise ValueError("Unsupported format {}".format(input_format))


def hflip(image, target, image_set='train'):
    flipped_image = F.hflip(image)
    target = target.copy()
    if image_set in ['test']:
        return flipped_image, target

    w, h = image.size
    if "boxes_h" in target:
        boxes = target["boxes_h"]
        boxes = boxes[:, [2, 1, 0, 3]] * \
            torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes_h"] = boxes
    if "boxes_o" in target:
        boxes = target["boxes_o"]
        boxes = boxes[:, [2, 1, 0, 3]] * \
            torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes_o"] = boxes
    if "action_boxes" in target:
        boxes = target["action_boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * \
            torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["action_boxes"] = boxes
    return flipped_image, target


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturatio=0, hue=0):
        self.color_jitter = tfs.ColorJitter(
            brightness, contrast, saturatio, hue)

    def __call__(self, img, target):
        return self.color_jitter(img), target


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target, image_set='train'):
        if random.random() < self.p:
            return hflip(img, target, image_set)
        return img, target


class RandomAdjustImage(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target, image_set='train'):
        if random.random() < self.p:
            img = F.adjust_brightness(
                img, random.choice([0.8, 0.9, 1.0, 1.1, 1.2]))
        if random.random() < self.p:
            img = F.adjust_contrast(
                img, random.choice([0.8, 0.9, 1.0, 1.1, 1.2]))
        return img, target


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """

    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target, image_set='train'):
        if random.random() < self.p:
            return self.transforms1(img, target, image_set)
        return self.transforms2(img, target, image_set)


def resize(image, target, size, max_size=None, image_set='train'):
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(
                    round(
                        max_size *
                        min_original_size /
                        max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return h, w
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return oh, ow

    rescale_size = get_size_with_aspect_ratio(
        image_size=image.size, size=size, max_size=max_size)
    rescaled_image = F.resize(image, rescale_size)

    if target is None:
        return rescaled_image, None
    target = target.copy()
    if image_set in ['test']:
        return rescaled_image, target

    ratios = tuple(float(s) / float(s_orig)
                   for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    if "boxes_h" in target:
        boxes = target["boxes_h"]
        scaled_boxes = boxes * \
            torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes_h"] = scaled_boxes
    if "boxes_o" in target:
        boxes = target["boxes_o"]
        scaled_boxes = boxes * \
            torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes_o"] = scaled_boxes
    if "action_boxes" in target:
        boxes = target["action_boxes"]
        scaled_boxes = boxes * \
            torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["action_boxes"] = scaled_boxes
    if "keypoint" in target:
        keypoint = target["keypoint"]
        scaled_keypoint = keypoint * \
            torch.as_tensor([ratio_width, ratio_height])
        target["keypoint"] = scaled_keypoint
    return rescaled_image, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None, image_set='train'):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size, image_set)


def crop(image, org_target, region, image_set='train'):
    cropped_image = F.crop(image, *region)
    target = org_target.copy()
    if image_set in ['test']:
        return cropped_image, target

    i, j, h, w = region
    fields = ["labels", "object", "actions"]

    if "boxes_h" in target:
        boxes = target["boxes_h"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        target["boxes_h"] = cropped_boxes.reshape(-1, 4)
        fields.append("boxes_h")
    if "boxes_o" in target:
        boxes = target["boxes_o"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        target["boxes_o"] = cropped_boxes.reshape(-1, 4)
        fields.append("boxes_o")
    if "action_boxes" in target:
        boxes = target["action_boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        target["action_boxes"] = cropped_boxes.reshape(-1, 4)
        fields.append("action_boxes")

    # remove elements for which the boxes or masks that have zero area
    if "boxes_h" in target and "boxes_o" in target:
        cropped_boxes = target['boxes_h'].reshape(-1, 2, 2)
        keep1 = torch.all(cropped_boxes[:, 1, :]
                          > cropped_boxes[:, 0, :], dim=1)
        cropped_boxes = target['boxes_o'].reshape(-1, 2, 2)
        keep2 = torch.all(cropped_boxes[:, 1, :]
                          > cropped_boxes[:, 0, :], dim=1)
        keep = keep1 * keep2
        if keep.any().sum() == 0:
            return image, org_target
        for field in fields:
            target[field] = target[field][keep]
    return cropped_image, target


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict, image_set='train'):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = tfs.RandomCrop.get_params(img, (h, w))
        return crop(img, target, region, image_set)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target, image_set='train'):
        image = torchvision.transforms.functional.normalize(
            image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        if image_set in ['test']:
            return image, target
        h, w = image.shape[-2:]
        if "human_boxes" in target:
            boxes = target["human_boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["human_boxes"] = boxes
        if "object_boxes" in target:
            boxes = target["object_boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["object_boxes"] = boxes
        if "action_boxes" in target:
            boxes = target["action_boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["action_boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, image_set='train'):
        for t in self.transforms:
            if isinstance(t, ToTensor):
                # image = t((image, target))
                image = t(image)
                return image, target
            elif isinstance(t, ColorJitter):
                image, target = t(image, target)
            else:
                image, target = t(image, target, image_set)
        return image, target


class ToTensor:
    """
    Convert to tensor, supporting below type:
        - tensor: torch.Tensor
        - list: List
        - dict: Dict
        - tuple: Tuple
        - pil: PIL
        - comb: special type, e.g. (PIL, Dict)
    """

    def __init__(self, input_format='tensor', dtype=None, device=None):
        self.input_format = input_format
        self.dtype = dtype
        self.device = device

    def __call__(self, x):
        return to_tensor(x,
                         input_format=self.input_format,
                         dtype=self.dtype,
                         device=self.device
                         )

    def __repr__(self):
        reprstr = self.__class__.__name__ + '('
        reprstr += 'input_format={}'.format(repr(self.input_format))
        reprstr += ', dtype='
        reprstr += repr(self.dtype)
        reprstr += ', device='
        reprstr += repr(self.device)
        reprstr += ')'
        return reprstr
