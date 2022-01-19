from .hico import build as build_hico
import easydict


def build_dataset(image_set: str, config: easydict.EasyDict, test_scale=-1):
    assert config.DATASET.NAME in [
        'hico', 'vcoco', 'hoia'], config.DATASET.NAME
    if config.DATASET.NAME == 'hico':
        return build_hico(image_set, test_scale)
