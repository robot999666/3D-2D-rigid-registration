# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import torch
import torch.nn as nn

from .network_modules import ConvBNReLU, ResnetBasicBlock, PositionalEmbedding

class RefineNet(nn.Module):
    def __init__(self, cfg=None, c_in=3, n_view=1):
        """A/B 共享编码器，回归相对平移和旋转。

        本项目调用 c_in=5：三通道轮廓、深度和肝脏掩码。"""
        super().__init__()
        self.cfg = cfg

        use_BN = cfg.use_BN if hasattr(cfg, 'use_BN') else cfg.get('use_BN', True)
        rot_rep = cfg.rot_rep if hasattr(cfg, 'rot_rep') else cfg.get('rot_rep', '6d')

        if use_BN:
            norm_layer = nn.BatchNorm2d
            norm_layer1d = nn.BatchNorm1d
        else:
            norm_layer = None
            norm_layer1d = None

        # A/B 共享特征编码器。
        self.encodeA = nn.Sequential(
            ConvBNReLU(C_in=c_in, C_out=64, kernel_size=7, stride=2, norm_layer=norm_layer),
            ConvBNReLU(C_in=64, C_out=128, kernel_size=3, stride=2, norm_layer=norm_layer),
            ResnetBasicBlock(128, 128, bias=True, norm_layer=norm_layer),
            ResnetBasicBlock(128, 128, bias=True, norm_layer=norm_layer),
        )

        # 融合 A/B 特征。
        self.encodeAB = nn.Sequential(
            ResnetBasicBlock(256, 256, bias=True, norm_layer=norm_layer),
            ResnetBasicBlock(256, 256, bias=True, norm_layer=norm_layer),
            ConvBNReLU(256, 512, kernel_size=3, stride=2, norm_layer=norm_layer),
            ResnetBasicBlock(512, 512, bias=True, norm_layer=norm_layer),
            ResnetBasicBlock(512, 512, bias=True, norm_layer=norm_layer),
        )

        embed_dim = 512
        num_heads = 4

        self.feature_token_grid = self._parse_feature_token_grid(
            os.environ.get("AR_FEATURE_TOKEN_GRID", "")
        )
        if self.feature_token_grid is None:
            self.feature_token_pool = None
            max_len = int(os.environ.get("AR_POS_EMBED_MAX_LEN", "10000"))
        else:
            grid_w, grid_h = self.feature_token_grid
            self.feature_token_pool = nn.AdaptiveAvgPool2d((grid_h, grid_w))
            max_len = grid_w * grid_h

        # 可选池化在 Transformer 前固定 token 数。
        self.pos_embed = PositionalEmbedding(d_model=embed_dim, max_len=max_len)

        self.trans_head = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
            nn.Linear(512, 3),
        )

        if rot_rep == 'axis_angle':
            rot_out_dim = 3
        elif rot_rep == '6d':
            rot_out_dim = 6
        else:
            raise RuntimeError(f"Unsupported rot_rep: {rot_rep}")

        self.rot_head = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
            nn.Linear(512, rot_out_dim),
        )

    @staticmethod
    def _parse_feature_token_grid(text):
        text = str(text or "").strip().lower()
        if not text:
            return None
        for sep in ("x", ","):
            if sep in text:
                left, right = text.split(sep, 1)
                grid_w, grid_h = int(left), int(right)
                if grid_w <= 0 or grid_h <= 0:
                    raise ValueError("AR_FEATURE_TOKEN_GRID dimensions must be positive.")
                return grid_w, grid_h
        raise ValueError("AR_FEATURE_TOKEN_GRID must look like '64x36' or '64,36'.")

    def forward(self, A, B):
        """A、B 形状均为 (batch, c_in, height, width)。"""
        bs = len(A)
        output = {}

        # 合并 batch，共用一次特征提取。
        x = torch.cat([A, B], dim=0)
        x = self.encodeA(x)

        a = x[:bs]
        b = x[bs:]

        ab = torch.cat((a, b), dim=1).contiguous()
        ab = self.encodeAB(ab)
        if self.feature_token_pool is not None:
            ab = self.feature_token_pool(ab)

        # 空间特征展平为 token 序列。
        ab = ab.reshape(bs, ab.shape[1], -1).permute(0, 2, 1)
        ab = self.pos_embed(ab)

        output['trans'] = self.trans_head(ab).mean(dim=1)
        output['rot'] = self.rot_head(ab).mean(dim=1)

        return output
