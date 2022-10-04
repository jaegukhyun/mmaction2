from mmaction.models.backbones import *
from mmaction.models.necks import *

import torch


def get_slowfast():
    model = ResNet3dSlowFast(pretrained=False)
    return model

def get_x3d():
    model = X3D()
    return model

def get_mvit():
    model =  MViT(
        spatial_size=256,
        num_frames=32,
        enable_detection=True,
        zero_decay_pos_cls=False,
        use_abs_pos=False,
        rel_pos_spatial=True,
        rel_pos_temporal=True,
        depth=16,
        num_heads=1,
        embed_dim=96,
        patch_kernel=(3, 7, 7),
        patch_stride=(2, 4, 4),
        patch_padding=(1, 3, 3),
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.2,
        norm="layernorm",
        mode="conv",
        cls_embed_on=True,
        _dim_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
        _head_mul=[[1, 2.0], [3, 2.0], [14, 2.0]],
        pool_kvq_kernel=[3, 3, 3],
        pool_kv_stride_adaptive=[1, 8, 8],
        pool_q_stride=[[0, 1, 1, 1], [1, 1, 2, 2], [2, 1, 1, 1], [3, 1, 2, 2],
                      [4, 1, 1, 1], [5, 1, 1, 1], [6, 1, 1, 1], [7, 1, 1, 1],
                      [8, 1, 1, 1], [9, 1, 1, 1], [10, 1, 1, 1], [11, 1, 1, 1],
                      [12, 1, 1, 1], [13, 1, 1, 1], [14, 1, 2, 2], [15, 1, 1, 1]],
        dropout_rate=0.0,
        dim_mul_in_att=True,
        residual_pooling=True,
    )
    return model

def get_movinet():
    model = MoViNet(name='MoViNetA0')
    return model

def get_yowo():
    from mmdet.models.backbones import CSPDarknet

    backbone_2d = CSPDarknet(deepen_factor=0.33, widen_factor=0.375)
    backbone_3d = ResNet3dSlowFast(pretrained=None)

    model = YOWOBackbone(backbone_2d, backbone_3d, 32)
    return model

def main():
    rinput = torch.randn(4, 3, 32, 256, 256)

    print('SlowFast output shape')
    model = get_slowfast()
    print(model(rinput)[0].shape, model(rinput)[1].shape)

    print('X3D output shape')
    model = get_x3d()
    print(model(rinput).shape)

    print('MViTv2 output shape')
    model = get_mvit()
    print(model(rinput).shape)

    print('MoViNet output shape')
    model = get_movinet()
    print(model(rinput).shape)

    print('YOWO output shape')
    model = get_yowo()
    print(model(rinput).shape)

if __name__ == '__main__':
    main()

