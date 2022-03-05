import copy
import math
import torch.nn.functional as F

import torch
from torch import nn


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class SemanticGraph(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            num_layers,
            attention_type='embedded_dot_pro',
            head_num=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attention_type = attention_type

        if attention_type == 'embedded_dot_pro':
            self.relation_dim = hidden_dim
            self.semantic_q = [nn.Linear(input_dim, self.relation_dim), ]
            self.semantic_k = [nn.Linear(input_dim, self.relation_dim), ]
            self.semantic_v = [nn.Linear(input_dim, hidden_dim), ]
            self.semantic_proj_res = nn.Linear(input_dim, hidden_dim)
            for _ in range(num_layers - 1):
                self.semantic_q.append(nn.Linear(hidden_dim, hidden_dim))
                self.semantic_k.append(nn.Linear(hidden_dim, hidden_dim))
                self.semantic_v.append(nn.Linear(hidden_dim, hidden_dim))
            self.semantic_q = nn.ModuleList(self.semantic_q)
            self.semantic_k = nn.ModuleList(self.semantic_k)
            self.semantic_v = nn.ModuleList(self.semantic_v)

        elif attention_type == 'multihead_transformer':
            assert self.num_layers == 1
            self.head_num = head_num
            self.bottleneck_dim = int(self.hidden_dim // 0.5)
            self.relation_dim = hidden_dim // self.head_num
            self.semantic_q = nn.Linear(input_dim, self.relation_dim)
            self.semantic_k = nn.Linear(input_dim, self.relation_dim)
            self.semantic_q = _get_clones(self.semantic_q, self.head_num)
            self.semantic_k = _get_clones(self.semantic_k, self.head_num)
            self.semantic_v = nn.Linear(input_dim, self.relation_dim)
            # self.coef = nn.ParameterList([nn.Parameter(torch.ones((hidden_dim, ), dtype = torch.float)/math.sqrt(hidden_dim), requires_grad = True) for _ in range(self.head_num)])
            self.semantic_proj_res = nn.Linear(input_dim, hidden_dim)

            self.W_t2 = nn.Linear(hidden_dim, self.bottleneck_dim)
            self.dropout2 = nn.Dropout(0.1)
            self.W_t1 = nn.Linear(self.bottleneck_dim, hidden_dim)
            self.LayerNorm = nn.LayerNorm([self.bottleneck_dim, ])

        elif attention_type == 'MLP':
            self.mlp_layers = 3
            self.mlp = nn.ModuleList([nn.Linear(input_dim, hidden_dim) if i == 0 else nn.Linear(
                hidden_dim, hidden_dim) for i in range(self.mlp_layers)])
            # self.nonlinearity = nn.ModuleList([nn.LeakyReLU(negative_slope=0.2, inplace=False) for i in range(self.mlp_layers-1)])
            self.nonlinearity = nn.ModuleList(
                [nn.ReLU() for i in range(self.mlp_layers)])
            self.mlp_ln = nn.LayerNorm([hidden_dim, ])

        elif attention_type == 'MLP_GNN':
            self.mlp_layers = 2
            self.mlp = nn.ModuleList([nn.Linear(input_dim, hidden_dim) if i == 0 else nn.Linear(
                hidden_dim, hidden_dim) for i in range(self.mlp_layers)])
            # self.nonlinearity = nn.ModuleList([nn.LeakyReLU(negative_slope=0.2, inplace=False) for i in range(self.mlp_layers-1)])
            self.nonlinearity = nn.ModuleList(
                [nn.ReLU() for i in range(self.mlp_layers)])
            self.mlp_ln = nn.ModuleList(
                [nn.LayerNorm([hidden_dim, ]) for i in range(self.mlp_layers)])

            self.relation_dim = hidden_dim
            self.semantic_ln = nn.ModuleList(
                [nn.LayerNorm([hidden_dim, ]) for _ in range(num_layers)])
            self.semantic_nonlinear = nn.ModuleList(
                [nn.ReLU() for _ in range(num_layers)])
            self.semantic_q = nn.ModuleList(
                [nn.Linear(hidden_dim, self.relation_dim) for _ in range(num_layers)])
            self.semantic_k = nn.ModuleList(
                [nn.Linear(hidden_dim, self.relation_dim) for _ in range(num_layers)])
            self.semantic_v = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])

            # Bilinear Pooling
            # self.nheads = nheads
            # self.bilinear1 = nn.Linear(hidden_dim, hidden_dim)
            # self.bilinear2 = nn.Linear(hidden_dim, hidden_dim)
            # self.bilinear1 = _get_clones(self.bilinear1, nheads)
            # self.bilinear2 = _get_clones(self.bilinear2, nheads)
            # self.coef = nn.ParameterList([nn.Parameter(torch.ones((hidden_dim, ), dtype = torch.float)/math.sqrt(hidden_dim), requires_grad = True) for _ in range(nheads)])

            # hid_hid_dim = hidden_dim//nheads
            # self.W3 = nn.Linear(hidden_dim, hid_hid_dim)
            # self.W3 = _get_clones(self.W3, nheads)
            # self.W2 = nn.Linear(hidden_dim, hidden_dim)
            # self.W1 = nn.Linear(hidden_dim, hidden_dim)
            # self.nonlinear = nn.ReLU(inplace = True)
            # self.LayerNorm = nn.LayerNorm([hidden_dim,])

    def forward(self, x, cooccur_prior=None):
        assert len(x.shape) == 2
        if self.attention_type == 'embedded_dot_pro':
            for i in range(self.num_layers):
                x_q = self.semantic_q[i](x)
                x_k = self.semantic_k[i](x)
                x_v = self.semantic_v[i](x)
                # x_att = torch.einsum('ac,bc->ab', x_q, x_k)
                x_att = torch.einsum('ac,bc->ab', x_q, x_k) / \
                    math.sqrt(self.relation_dim)
                x_att = F.softmax(x_att, dim=-1)
                if cooccur_prior is not None:
                    x_att = x_att + cooccur_prior
                    print('cooccur prior')

                if i == 0:
                    # self.verb_calibration_embedding
                    x = F.relu(torch.matmul(x_att, x_v)) + \
                        self.semantic_proj_res(x)
                else:
                    x = F.relu(torch.matmul(x_att, x_v)) + x
            trans_x = x
            # trans_x = norm_tensor(x)

        if self.attention_type == 'multihead_transformer':
            len_x = len(x.shape)
            if len_x == 2:
                x = x.unsqueeze(dim=0)
            elif len_x == 4:
                l, bs, q, hiddim = x.shape
                x = x.view((l * bs, q, hiddim))
            elif len_x == 3:
                None
            else:
                print("Shape is not compatible")
                assert False

            x_v = self.semantic_v(x)
            multihead_ft = []
            for i in range(self.head_num):
                x_q_i = self.semantic_q[i](x)  # lbs, q, hiddim
                # * self.coef[i].expand_as(x_q_i)  # lbs, q, hiddim
                x_k_i = self.semantic_k[i](x)

                x_att_i = torch.einsum(
                    'abc,adc->abd', x_q_i, x_k_i) / math.sqrt(self.relation_dim)
                x_att_i = F.softmax(x_att_i, dim=-1)
                att_ft_i = torch.bmm(x_att_i, x_v)
                multihead_ft.append(att_ft_i)

            multihead_ft = torch.cat(multihead_ft, dim=-1)
            trans_ft = self.W_t1(
                F.relu(
                    self.LayerNorm(
                        self.W_t2(multihead_ft)),
                    inplace=True))
            trans_x = trans_ft + self.semantic_proj_res(x)

            if len_x == 2:
                trans_x = trans_x.squeeze(dim=0)
            elif len_x == 4:
                trans_x = trans_x.view((l, bs, q, hiddim))
            elif len_x == 3:
                None

        if self.attention_type == 'MLP':
            for i in range(self.mlp_layers):
                x = self.mlp[i](x)
                if i == self.mlp_layers - 1:
                    x = self.mlp_ln(x)
                    x = self.nonlinearity[i](x)
                else:
                    x = self.nonlinearity[i](x)

            # trans_x = norm_tensor(x)
            trans_x = x

        if self.attention_type == 'MLP_GNN':
            for i in range(self.mlp_layers):
                x = self.mlp[i](x)
                x = self.mlp_ln[i](x)
                x = self.nonlinearity[i](x)

            for i in range(self.num_layers):
                x_q = self.semantic_q[i](x)
                x_k = self.semantic_k[i](x)
                x_v = self.semantic_v[i](x)
                x_att = torch.einsum('ac,bc->ab', x_q, x_k) / \
                    math.sqrt(self.relation_dim)
                x_att = F.softmax(x_att, dim=-1)
                x = self.semantic_nonlinear[i](
                    self.semantic_ln[i](
                        torch.matmul(
                            x_att, x_v))) + x

            trans_x = x

        return trans_x
