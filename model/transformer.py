import easydict
from torch.nn import Transformer


def build_transformer(config: easydict):
    transformer = Transformer(config.MODEL.HIDDEN_DIM,
                              config.MODEL.NHEAD,
                              config.MODEL.ENC_LAYERS,
                              config.MODEL.DEC_LAYERS,
                              config.MODEL.DIM_FEEDFORWARD,
                              config.MODEL.DROPOUT)
    return transformer
