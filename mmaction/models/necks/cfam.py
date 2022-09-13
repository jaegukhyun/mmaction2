import torch
import torch.nn as nn

from ..builder import NECKS

try:
    from mmdet.models import NECKS as MMDET_NECKS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


class CAM_Module(nn.Module):
    """ Channel attention module """
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W )
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CFAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CFAMBlock, self).__init__()
        inter_channels = 1024
        self.conv_bn_relu1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels,
                                                     kernel_size=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv_bn_relu2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels,
                                                     3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.sc = CAM_Module(inter_channels)

        self.conv_bn_relu3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels,
                                                     3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.ReLU())

        self.conv_out = nn.Sequential(nn.Dropout2d(0.1, False),
                                      nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):

        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.sc(x)
        x = self.conv_bn_relu3(x)
        output = self.conv_out(x)

        return output


@NECKS.register_module()
class CFAM(nn.Module):
    def __init__(self,
                 channels_2d,
                 channels_3d,
                 out_channels,
                 neck_2d=None,
                 neck_3d=None):
        super().__init__()
        self.neck_2d = MMDET_NECKS.build(neck_2d)
        self.neck_3d = NECKS.build(neck_3d)
        self.block = CFAMBlock(channels_2d+channels_3d, out_channels)

    def forward(self, x):
        feature_3d, feature_2d = x

        if self.neck_2d is not None:
            feature_2d = self.neck_2d(feature_2d)
        if self.neck_3d is not None:
            feature_3d = self.neck_3d(feature_3d)

        if isinstance(feature_2d, tuple):
            feature_2d = feature_2d[-1]
        if feature_3d.shape[2] > 1:
            feature_3d = torch.mean(feature_3d, 2, keepdim=True)

        feature_3d = torch.squeeze(feature_3d, dim=2)

        f = torch.cat((feature_3d, feature_2d), dim=1)
        output = self.block(f)

        return [output]


if mmdet_imported:
    MMDET_NECKS.register_module()(CFAM)
