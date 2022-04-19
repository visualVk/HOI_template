from torch import nn
from model.asnet import MLP
from model.transformer import TransformerDecoder, TransformerDecoderLayer


class Simple_Joint(nn.Module):
    def __init__(self, hidden_dim, num_joints) -> None:
        super(Simple_Joint, self).__init__()
        self.num_joints = num_joints
        self.preprocess = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.linear = nn.Linear(hidden_dim, num_joints * 2)

    def forward(self, x):
        repr = self.preprocess(x)
        x = self.linear(x)
        return repr, x.view(x.size(0), self.num_joints, 2)

def build_simple_joint(hidden_dim, num_joint):
    net = Simple_Joint(hidden_dim, num_joint)
    return net