import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from model.lib import ST_RenovateNet

from transformer.transformer import Transformer
from transformer.cross_transformer import EncoderLayer


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding  
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList() 
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)

            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))
            A1 = A1 + A[i]
            A2 = x.reshape(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class TransformerOnT(nn.Module):
    def __init__(self, in_channel, out_channel, stride, d_model=512, max_len=100,
                 n_layers=1, n_head=8, drop_prob=0.1, device='cuda',
                 src_pad_idx=0):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(d_model)
        self.transformer = Transformer(src_pad_idx=src_pad_idx,
                                       d_model=d_model,
                                       n_head=n_head,
                                       max_len=max_len,
                                       ffn_hidden=2 * d_model,
                                       n_layers=n_layers,
                                       drop_prob=drop_prob,
                                       device=device)
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        N, C, T, V = x.size()

        x_T = x.permute(0, 3, 1, 2).reshape(N * V, C, T)
        # x_T = self.norm1(x_T)
        x_T = x_T.permute(0, 2, 1)
        out_T = self.transformer(x_T) 
        out_T = out_T.view(N, V, T, C).permute(0, 3, 2, 1) 
        out_T = self.bn(self.conv(out_T))
        return out_T


class CGT_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(CGT_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        self.transformer = TransformerOnT(in_channel=out_channels, out_channel=out_channels, stride=stride,
                                          d_model=out_channels)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1))

    def forward(self, x):
        y = self.gcn1(x)

        x = self.transformer(y) + self.residual(x)
        return self.relu(x)


class SpatialAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(SpatialAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = avg_out
        return self.sigmoid(out)


class fusionOnT(nn.Module):
    def __init__(self, d_model):
        super(fusionOnT, self).__init__()
        self.transformer = EncoderLayer(d_model, 8, 2 * d_model, 0.1)

    def forward(self, x, y):
        x_, y_ = self.transformer(x, y)
        x, y = x + x_, y + y_
        return x, y


class fusionOnV(nn.Module):
    def __init__(self):
        super(fusionOnV, self).__init__()

        self.att_N_p = SpatialAttention(16)  
        self.att_N_m = SpatialAttention(16)  

    def forward(self, x_p, x_m):

        B, C, T, N = x_p.size()

        x_p_N = x_p.permute(0, 3, 2, 1)
        att_N_p_map = self.att_N_p(x_p_N).permute(0, 3, 2, 1)

        x_m_N = x_m.permute(0, 3, 2, 1)
        att_N_m_map = self.att_N_m(x_m_N).permute(0, 3, 2, 1)

        x_p_new = x_p + (x_m* att_N_m_map)
        x_m_new = x_m + (x_p* att_N_p_map)
        return x_p_new, x_m_new


class BCSF(nn.Module):
    def __init__(self,d_model):
        super(BCSF, self).__init__()
        self.fusionT = fusionOnT(d_model)
        self.fusionV = fusionOnV()
    def forward(self,x,y):
        x,y = self.fusionT(x,y)
        x,y = self.fusionV(x,y)
        return x,y


        
class Model(nn.Module):
    def __init__(self, num_class=4, num_point=16, num_constraints=31, graph=None, graph_args=dict(), in_channels_p=3,
                 in_channels_m=8, drop_out=0, cl_mode=None, multi_cl_weights=[1, 1, 1, 1], cl_version='V0', **kwargs):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A

        self.cl_mode = cl_mode
        self.multi_cl_weights = multi_cl_weights
        self.cl_version = cl_version

        self.data_bn_p = nn.BatchNorm1d(in_channels_p * num_point)
        self.data_bn_m = nn.BatchNorm1d(in_channels_m * num_point)

        self.l1_p = CGT_unit(in_channels_p, 64, A, residual=False)
        self.l1_m = CGT_unit(in_channels_m, 64, A, residual=False)

        self.l2_p = CGT_unit(64, 64, A)
        self.l5_p = CGT_unit(64, 128, A, stride=2)
        self.l8_p = CGT_unit(128, 256, A, stride=2)

        self.l2_m = CGT_unit(64, 64, A)
        self.l5_m = CGT_unit(64, 128, A, stride=2)
        self.l8_m = CGT_unit(128, 256, A, stride=2)

        self.fusion = BCSF(64)

        self.fc1_classifier_p = nn.Linear(256, num_class)
        self.fc1_classifier_m = nn.Linear(256, num_class)
        self.fc2_aff = nn.Linear(256, num_constraints * 48)

        nn.init.normal_(self.fc1_classifier_m.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.fc1_classifier_p.weight, 0, math.sqrt(2. / num_class))
        nn.init.normal_(self.fc2_aff.weight, 0, math.sqrt(2. / (num_constraints * 48)))
        bn_init(self.data_bn_p, 1)
        bn_init(self.data_bn_m, 1)

        if self.cl_mode is not None:
            self.build_cl_blocks()

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def build_cl_blocks(self):
        if self.cl_mode == "ST-Multi-Level":
            self.ren_low = ST_RenovateNet(64, 48, 16, 1, n_class=4, version=self.cl_version)
            self.ren_mid = ST_RenovateNet(64, 48, 16, 1, n_class=4, version=self.cl_version)
            self.ren_high = ST_RenovateNet(128, 24, 16, 1, n_class=4, version=self.cl_version)
            self.ren_fin = ST_RenovateNet(256, 12, 16, 1, n_class=4, version=self.cl_version)
        else:
            raise KeyError(f"no such Contrastive Learning Mode {self.cl_mode}")

    def get_ST_Multi_Level_cl_output_p(self, x, feat_low, feat_mid, feat_high, feat_fin, label):
        logits = self.fc1_classifier_p(x)
        cl_low = self.ren_low(feat_low, label.detach(), logits.detach())
        cl_mid = self.ren_mid(feat_mid, label.detach(), logits.detach())
        cl_high = self.ren_high(feat_high, label.detach(), logits.detach())
        cl_fin = self.ren_fin(feat_fin, label.detach(), logits.detach())
        cl_loss_p = cl_low * self.multi_cl_weights[0] + cl_mid * self.multi_cl_weights[1] + \
                    cl_high * self.multi_cl_weights[2] + cl_fin * self.multi_cl_weights[3]
        return cl_loss_p

    def get_ST_Multi_Level_cl_output_m(self, x, feat_low, feat_mid, feat_high, feat_fin, label):
        logits = self.fc1_classifier_m(x)
        cl_low = self.ren_low(feat_low, label.detach(), logits.detach())
        cl_mid = self.ren_mid(feat_mid, label.detach(), logits.detach())
        cl_high = self.ren_high(feat_high, label.detach(), logits.detach())
        cl_fin = self.ren_fin(feat_fin, label.detach(), logits.detach())
        cl_loss_m = cl_low * self.multi_cl_weights[0] + cl_mid * self.multi_cl_weights[1] + \
                    cl_high * self.multi_cl_weights[2] + cl_fin * self.multi_cl_weights[3]
        return cl_loss_m

    def forward(self, x_p, x_m, label=None, get_cl_loss=False, get_hidden_feat=False, **kwargs):
        N, C_p, T, V, M = x_p.size()
        N, C_m, T, V, M = x_m.size()

        x_p = x_p.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C_p, T)
        x_m = x_m.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C_m, T)

        x_p = self.data_bn_p(x_p)
        x_m = self.data_bn_m(x_m)

        x_p = x_p.view(N, M, V, C_p, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C_p, T, V)
        x_m = x_m.view(N, M, V, C_m, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C_m, T, V)

        x_p = self.l1_p(x_p)
        feat_low_p = x_p.clone()
        x_m = self.l1_m(x_m)
        feat_low_m = x_m.clone()
        x_p = self.l2_p(x_p)
        feat_mid_p = x_p.clone()
        x_m = self.l2_m(x_m)
        feat_mid_m = x_m.clone()

        x_m, x_p = self.fusion(x_m, x_p)

        x_p = self.l5_p(x_p)
        feat_high_p = x_p.clone()
        x_m = self.l5_m(x_m)
        feat_high_m = x_m.clone()

        x_p = self.l8_p(x_p)
        feat_fin_p = x_p.clone()
        x_m = self.l8_m(x_m)
        feat_fin_m = x_m.clone()

        # N*M,C,T,V
        c_new_m = x_m.size(1)
        x_m = x_m.reshape(N, M, c_new_m, -1)
        x_m = x_m.mean(3).mean(1)

        c_new_p = x_p.size(1)
        x_p = x_p.reshape(N, M, c_new_p, -1)
        x_p = x_p.mean(3).mean(1)


        # x_cat=torch.cat((x_m,x_p),1)
        if get_cl_loss and self.cl_mode == "ST-Multi-Level":
            return (self.fc1_classifier_p(x_p), self.fc2_aff(x_p), self.fc1_classifier_m(x_m),
                    self.get_ST_Multi_Level_cl_output_p(x_p, feat_low_p, feat_mid_p, feat_high_p, feat_fin_p, label),
                    self.get_ST_Multi_Level_cl_output_m(x_m, feat_low_m, feat_mid_m, feat_high_m, feat_fin_m, label)
                    )

        return self.fc1_classifier_p(x_p), self.fc2_aff(x_p), self.fc1_classifier_m(x_m)

