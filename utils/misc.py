import argparse
import random
from typing import List, Optional

import easydict
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from torch import Tensor


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    @property
    def device(self):
        return self.mask.device

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    """
    convert a batch of images with different shape into unified image
    Args:
        tensor_list: the almost case is a batch of images

    Returns:
        NestedTensor: [tensor, mask], mask is designed for representing original tensor in resized tensor

    """
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        # get max size of image [c, h, w]
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        # size: [b, max_c, max_h, max_w]
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        # after croping image size
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            # if original image size less than cropped size, transformer should focus on original region
            # so we need to create mask for every batch
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list):
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack(
            [img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(
            img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(
            m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class AverageMeter(object):
    def __init__(self, name):
        self.name = name
        self.reset()

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self._avg = self.sum / self.count

    def reset(self):
        self.val = 0
        self.sum = 0
        self._avg = 0
        self.count = 1

    def avg(self):
        return self._avg

    def global_avg(self):
        return self.sum / self.count

    def synchronize_between_process(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.sum],
                         dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.sum = t[1]

    def __str__(self):
        fmt = "{}: val = {:.5f}, avg = {:.5f}".format(
            self.name, self.val, self._avg)
        return fmt


# def collate_fn(batch):
#     batch = list(zip(*batch))
#     batch[0] = nested_tensor_from_tensor_list(batch[0])
#     return tuple(batch)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args: argparse.Namespace):
    torch.cuda.set_device(args.local_rank)

    dist.init_process_group(
        backend=args.backend,
        init_method="env://",
        world_size=args.world_size,
        rank=args.local_rank)


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def reduce_loss(loss: nn.Module, average=True):
    world_size = dist.get_world_size()
    if world_size < 2:
        return loss
    with torch.no_grad:
        reduce_loss = loss
        dist.all_reduce(reduce_loss)
        if average:
            reduce_loss /= world_size
    return reduce_loss


def fix_random_seed(args: argparse.Namespace, config: easydict):
    if args.local_rank is not None:
        seed = config.SEED + args.local_rank
    else:
        seed = config.SEED + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def write_dict_to_json(mydict, f_path):
    import json
    import numpy

    class DateEnconding(json.JSONEncoder):
        def default(self, obj):
            if isinstance(
                obj,
                (numpy.int_,
                 numpy.intc,
                 numpy.intp,
                 numpy.int8,
                 numpy.int16,
                 numpy.int32,
                 numpy.int64,
                 numpy.uint8,
                 numpy.uint16,
                 numpy.uint32,
                 numpy.uint64)):
                return int(obj)
            elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
                                  numpy.float64)):
                return float(obj)
            elif isinstance(obj, (numpy.ndarray,)):  # add this line
                return obj.tolist()  # add this line
            return json.JSONEncoder.default(self, obj)
    with open(f_path, 'w') as f:
        json.dump(mydict, f, cls=DateEnconding)
        print("write down det dict to %s!" % (f_path))
