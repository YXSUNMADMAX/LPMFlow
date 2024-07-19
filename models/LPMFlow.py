import os
from functools import reduce, partial
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath
from models.feature_backbones import vision_transformer
from models.mod import FeatureL2Norm, unnormalise_and_convert_mapping_to_flow
from models.base.swin import SwinTransformer2d, TransformerWarper2d
from models.encoder.encoder.single_tower_smask import SingleTower
import random
import torch
from torch import nn

r"""
Modified timm library Vision Transformer implementation
https://github.com/rwightman/pytorch-image-models
"""


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        x = x + x * out
        return x

class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def softmax_with_temperature(self, x, beta, d=1):
        r"""SFNet: Learning Object-aware Semantic Flow (Lee et al.)"""
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M  # subtract maximum value for stability
        exp_x = torch.exp(x / beta)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum

    def forward(self, x, y, mode):

        if mode == "self":
            B, N, C = x.shape
            qkv = (
                self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )  # make torchscript happy (cannot use tensor as tuple)
            # print('q',q.shape)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x, attn
        else:
            B, Nx, C = x.shape
            _, Ny, _ = y.shape
            qkvx = (
                self.qkv(x)
                .reshape(B, Nx, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            qx, kx, vx = qkvx[0], qkvx[1], qkvx[2]

            qkvy = (
                self.qkv(y)
                .reshape(B, Ny, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            qy, ky, vy = qkvy[0], qkvy[1], qkvy[2]

            attn = (qx @ ky.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ vy).transpose(1, 2).reshape(B, Nx, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x, attn.sum(1)

class MultiscaleDecoder(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            last=False,
    ):
        super().__init__()

        self.sattn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.cattn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.last = last
        self.gelu = nn.GELU()
        self.pos_embed_src = nn.Parameter(torch.zeros(1, 256 + 1, 768))
        self.pos_embed_trg = nn.Parameter(torch.zeros(1, 256 + 1, 768))
        self.pos_embed_src_up = nn.Parameter(torch.zeros(1, 1024 + 1, 768))
        self.pos_embed_trg_up = nn.Parameter(torch.zeros(1, 1024 + 1, 768))

    def arf(self, x):
        exp = torch.exp
        zero = torch.zeros_like(x)
        tmp = (exp(x) - exp(-1 * x)) / (exp(x) + exp(-1 * x - 4))
        return torch.where(tmp < 0, zero, tmp)

    def forward(self, ins):
        """
        Multi-level aggregation
        """
        src, tgt, attm = ins
        B, N1, C = src.shape
        _, N2, _ = tgt.shape

        if N1 > 258:
            src = src + self.pos_embed_src_up
        else:
            src = src + self.pos_embed_src
        if N2 > 258:
            tgt = tgt + self.pos_embed_trg_up
        else:
            tgt = tgt + self.pos_embed_trg

        srct, _ = self.sattn(self.norm1(src), None, "self")
        srct = self.arf(srct)
        src = src + self.drop_path(srct)
        src = src + self.drop_path(self.norm2(src))

        tgtt, _ = self.sattn(self.norm1(tgt), None, "self")
        tgtt = self.arf(tgtt)
        tgt = tgt + self.drop_path(tgtt)
        tgt = tgt + self.drop_path(self.mlp(self.norm2(tgt)))

        srct, attn_src = self.cattn(self.norm1(src), self.norm1(tgt), "cross")
        srct = self.arf(srct)
        src = src + self.drop_path(srct)
        src = src + self.drop_path(self.mlp(self.norm2(src)))
        attm.append(attn_src[:, 1:, 1:])
        return src.contiguous().view(B, N1, C), tgt.contiguous().view(B, N2, C), attm

class TransformerDecoder(nn.Module):
    def __init__(
            self,
            embed_dim=2048,
            depth_de=12,
            num_heads=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=None,
    ):
        super().__init__()
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth_de)
        ]  # stochastic depth decay rule

        self.blocks = nn.Sequential(
            *[
                MultiscaleDecoder(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    last=False,
                )
                for i in range(depth_de - 1)
            ]
        )

        self.last_block = nn.Sequential(
            *[
                MultiscaleDecoder(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[depth_de - 1],
                    norm_layer=norm_layer,
                    last=True,
                )
                for i in range(1)
            ]
        )

    def forward(self, src, tgt):
        src, tgt, attm = self.blocks((src, tgt, []))
        feat_src, feat_trg, corr_src = self.last_block((src, tgt, attm))
        return corr_src, feat_src, feat_trg


class FeatureExtractionHyperPixel_VIT(nn.Module):
    def __init__(self, feature_size, freeze=True, ibot_ckp_file='./'):
        super().__init__()

        self.backbone = vision_transformer.__dict__['vit_base'](patch_size=16, num_classes=0)
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        dino_ckp_file = ibot_ckp_file

        if os.path.isfile(dino_ckp_file):
            state_dict = torch.load(dino_ckp_file, map_location="cpu")
            dino_ckp_key = 'teacher'
            if dino_ckp_key is not None and dino_ckp_key in state_dict:
                print(f"Take key {dino_ckp_key} in provided checkpoint dict")
                state_dict = state_dict[dino_ckp_key]
            for k, v in state_dict.items():
                print(k)
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = self.backbone.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(dino_ckp_file, msg))

        self.feature_size = feature_size

    def forward(self, img):
        r"""Extract desired a list of intermediate features"""
        feat = self.backbone(img)
        return feat


class SRBranch(nn.Module):
    def __init__(self):
        super().__init__()

        self.refine_swin_decoder_16 = nn.Sequential(
            TransformerWarper2d(
                SwinTransformer2d(
                    img_size=(128, 128), embed_dim=96, window_size=16, num_heads=[8], depths=[1]
                )
            ),
            ChannelAttention(96),
        )

        self.refine_swin_decoder_8 = nn.Sequential(
            TransformerWarper2d(
                SwinTransformer2d(
                    img_size=(128, 128), embed_dim=96, window_size=8, num_heads=[8], depths=[1]
                )
            ),
            ChannelAttention(96),
        )

        self.refine_swin_decoder_4 = nn.Sequential(
            TransformerWarper2d(
                SwinTransformer2d(
                    img_size=(128, 128), embed_dim=96, window_size=4, num_heads=[8], depths=[1]
                )
            ),
            ChannelAttention(96),
        )

        self.refine2 = nn.Sequential(
            nn.Conv2d(96, 32, (3, 3), padding=(1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 2, (3, 3), padding=(1, 1), bias=True),
        )

    def forward(self, Reps):
        r"""Extract desired a list of intermediate features"""
        x_16 = self.refine_swin_decoder_16(Reps)
        x_8 = self.refine_swin_decoder_8(x_16)
        x_4 = self.refine_swin_decoder_4(x_8)
        x = x_16 + x_8 + x_4
        flow_refine = self.refine2(x)
        return flow_refine


class SRBranch_feat(nn.Module):
    def __init__(self):
        super().__init__()
        self.refine_swin_decoder_1_1 = nn.Sequential(
            TransformerWarper2d(
                SwinTransformer2d(
                    img_size=(32, 32), embed_dim=768, window_size=4, num_heads=[8], depths=[1]
                )
            ),
            ChannelAttention(768),
        )

        self.refine_swin_decoder_1_2 = nn.Sequential(
            TransformerWarper2d(
                SwinTransformer2d(
                    img_size=(32, 32), embed_dim=768, window_size=4, num_heads=[8], depths=[1]
                )
            ),
            ChannelAttention(768),
        )

    def forward(self, x):
        r"""Extract desired a list of intermediate features"""
        x_cls = x[:, 0:1, :]
        x_patch = x[:, 1:, :].transpose(-2, -1)
        B, N, C = x.shape
        x_patch = x_patch.view(B, C, 16, 16)
        x_patch = F.interpolate(x_patch, 32, None, 'bilinear', False)
        x_2 = self.refine_swin_decoder_1_1(x_patch)
        x_2 = self.refine_swin_decoder_1_2(x_2)
        x_2 = x_2.flatten(-2, -1).transpose(-2, -1)
        x_out = torch.cat([x_cls, x_2], dim=1)
        return x_out


class LPMFlow(nn.Module):
    def __init__(self, feature_size=128, ibot_ckp_file='./', depth=6, num_heads=8, mlp_ratio=4, freeze=False):
        super().__init__()
        self.feature_size = feature_size
        self.feature_size_model = 32
        self.decoder_embed_dim = 768
        self.feature_extraction = FeatureExtractionHyperPixel_VIT(feature_size, freeze, ibot_ckp_file)
        self.x_normal = np.linspace(-1, 1, self.feature_size_model)
        self.x_normal = nn.Parameter(
            torch.tensor(self.x_normal, dtype=torch.float, requires_grad=False)
        )
        self.y_normal = np.linspace(-1, 1, self.feature_size_model)
        self.y_normal = nn.Parameter(
            torch.tensor(self.y_normal, dtype=torch.float, requires_grad=False)
        )

        self.l2norm = FeatureL2Norm()

        self.encoder = SingleTower(embed_dim=self.decoder_embed_dim,
                                   depth=6,
                                   num_heads=num_heads,
                                   mlp_ratio=mlp_ratio,
                                   drop_rate=0.1,
                                   qkv_bias=True,
                                   norm_layer=partial(nn.LayerNorm, eps=1e-6))

        self.decoder_s1 = TransformerDecoder(
            embed_dim=self.decoder_embed_dim,
            depth_de=1,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=0.1,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        self.decoder_s2 = TransformerDecoder(
            embed_dim=self.decoder_embed_dim,
            depth_de=1,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=0.1,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        self.decoder_s3 = TransformerDecoder(
            embed_dim=self.decoder_embed_dim,
            depth_de=1,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=0.1,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        self.SRBranches = SRBranch()
        self.SRBranch_feat = SRBranch_feat()

        self.refine_proj_query_feat = nn.Sequential(
            nn.Conv2d(self.decoder_embed_dim, 88, 1), nn.ReLU()
        )
        self.dropout2d = nn.Dropout2d(p=0.5)
        self.masked_tokens = nn.Parameter(torch.zeros(257, 768), requires_grad=True)
        self.pos_embed_src = nn.Parameter(torch.zeros(1, 256 + 1, 768))

    def softmax_with_temperature(self, x, beta, d=1):
        r"""SFNet: Learning Object-aware Semantic Flow (Lee et al.)"""
        M, _ = x.max(dim=d, keepdim=True)
        x = x - M  # subtract maximum value for stability
        exp_x = torch.exp(x / beta)
        exp_x_sum = exp_x.sum(dim=d, keepdim=True)
        return exp_x / exp_x_sum

    def soft_argmax(self, corr, beta=0.02):
        r"""SFNet: Learning Object-aware Semantic Flow (Lee et al.)"""
        b, _, h, w = corr.size()

        corr = self.softmax_with_temperature(corr, beta=beta, d=1)
        corr = corr.view(-1, h, w, h, w)  # (target hxw) x (source hxw)

        grid_x = corr.sum(dim=1, keepdim=False)  # marginalize to x-coord.
        x_normal = self.x_normal.expand(b, w)
        x_normal = x_normal.view(b, w, 1, 1)
        grid_x = (grid_x * x_normal).sum(dim=1, keepdim=True)  # b x 1 x h x w

        grid_y = corr.sum(dim=2, keepdim=False)  # marginalize to y-coord.
        y_normal = self.y_normal.expand(b, h)
        y_normal = y_normal.view(b, h, 1, 1)
        grid_y = (grid_y * y_normal).sum(dim=1, keepdim=True)  # b x 1 x h x w

        return grid_x, grid_y

    def mutual_nn_filter(self, correlation_matrix):  # BSHWHW
        r"""Mutual nearest neighbor filtering (Rocco et al. NeurIPS'18)"""
        corr_src_max = torch.max(correlation_matrix, dim=3, keepdim=True)[0]
        corr_trg_max = torch.max(correlation_matrix, dim=2, keepdim=True)[0]
        corr_src_max[corr_src_max == 0] += 1e-30
        corr_trg_max[corr_trg_max == 0] += 1e-30

        corr_src = correlation_matrix / corr_src_max
        corr_trg = correlation_matrix / corr_trg_max

        return correlation_matrix * (corr_src * corr_trg)

    def token_replace(self, src_feat, trg_feat, K=64, P=0.25):
        B, _, _ = src_feat.shape
        srcs = []
        mask_indexs = []
        for b in range(B):
            with torch.no_grad():
                tmp_src = src_feat.clone()
                tmp_trg = trg_feat.clone()
                front_heat_src = self.l2norm(tmp_src[b, 1:, :]) @ \
                                 self.l2norm(tmp_src[b, 0:1, :]).transpose(-2, -1)
                front_heat_trg = self.l2norm(tmp_src[b, 1:, :]) @ \
                                 self.l2norm(tmp_trg[b, 0:1, :]).transpose(-2, -1)
                front_heat = front_heat_src[:, 0] + front_heat_trg[:, 0]
                cls_b = src_feat[b, 0:1, :].clone() + trg_feat[b, 0:1, :].clone()

            val, idx = torch.topk(front_heat, k=K)
            L = int(K * P)
            index = torch.LongTensor(random.sample(range(K), L)).to(idx.device)
            idx = idx + 1
            src_feat_b = src_feat[b].clone()
            src_feat_b[idx[index], :] = cls_b
            srcs.append(src_feat_b)
            mask_indexs.append(idx[index])
        src_feat = torch.stack(srcs)
        src_feat = src_feat + self.pos_embed_src
        mask_indexs = torch.stack(mask_indexs)
        return src_feat, mask_indexs

    def corr(self, src, trg):
        return src.flatten(2).transpose(-1, -2) @ trg.flatten(2)

    def apply_dropout(self, dropout, *feats):
        sizes = [x.shape[-2:] for x in feats]
        max_size = max(sizes)
        resized_feats = [F.interpolate(x, size=max_size, mode="nearest") for x in feats]

        channel_list = [x.size(1) for x in feats]
        feats = dropout(torch.cat(resized_feats, dim=1))
        feats = torch.split(feats, channel_list, dim=1)
        recoverd_feats = [
            F.interpolate(x, size=size, mode="nearest") for x, size in zip(feats, sizes)
        ]
        return recoverd_feats

    def feat_up(self, x):
        r"""Extract desired a list of intermediate features"""
        x_cls = x[:, 0:1, :]
        x_patch = x[:, 1:, :].transpose(-2, -1)
        B, N, C = x.shape
        x_patch = x_patch.view(B, C, 16, 16)
        x_patch = F.interpolate(x_patch, 32, None, 'bilinear', False)
        x_patch = x_patch.flatten(-2, -1).transpose(-2, -1)
        x_out = torch.cat([x_cls, x_patch], dim=1)
        return x_out

    def forward(self, source, target, is_train=False):
        Size = 128
        mask_indexs = None
        B, _, _, _ = source.shape
        source = F.interpolate(source, 256, None, 'bilinear', False)
        target = F.interpolate(target, 256, None, 'bilinear', False)
        src_feat = self.feature_extraction(source)
        tgt_feat = self.feature_extraction(target)

        if is_train:
            src_feat, mask_indexs = self.token_replace(src_feat, tgt_feat)

        tokens, attns = self.encoder(src_feat, tgt_feat)
        src_feat, tgt_feat = tokens[:, :257, :], tokens[:, 257:, :]
        corrs_11, en_src_feat, en_trg_feat = self.decoder_s1(src_feat, tgt_feat)

        en_src_feat_up = self.SRBranch_feat(en_src_feat)
        en_tgt_feat_up = self.SRBranch_feat(en_trg_feat)

        corrs_12, en_src_feat_12, en_trg_feat_up_12 = self.decoder_s2(en_src_feat, en_tgt_feat_up)
        corrs_21, en_src_feat_up_21, en_trg_feat_21 = self.decoder_s2(en_src_feat_up, en_trg_feat)

        en_src_feat_up_12 = self.SRBranch_feat(en_src_feat_12)
        en_trg_feat_up_21 = self.SRBranch_feat(en_trg_feat_21)

        en_src_feat_up = en_src_feat_up_12 + en_src_feat_up_21
        en_tgt_feat_up = en_trg_feat_up_12 + en_trg_feat_up_21

        corrs_22, en_src_feat, en_trg_feat = self.decoder_s3(en_src_feat_up, en_tgt_feat_up)

        refine_feat = self.refine_proj_query_feat(
            en_src_feat[:, 1:, :].transpose(-1, -2).view(B, -1, 32, 32)
        )
        refine_feat = F.interpolate(refine_feat, Size, None, "bilinear", True)
        refine_feat = self.apply_dropout(self.dropout2d, refine_feat)[0]

        corr_11 = corrs_11[-1]
        corr_11 = corr_11.view(B, -1, 16, 16)
        corr_11 = F.interpolate(corr_11, 32, None, "bilinear", True)
        corr_11 = corr_11.view(B, 256, 1024)
        corr_11 = corr_11.view(B, 16, 16, -1).permute(0, 3, 1, 2)
        corr_11 = F.interpolate(corr_11, 32, None, "bilinear", True)
        grid_x, grid_y = self.soft_argmax(corr_11)
        flow_11 = torch.cat((grid_x, grid_y), dim=1)
        flow_11 = unnormalise_and_convert_mapping_to_flow(flow_11)
        flow_up_11 = F.interpolate(flow_11, Size, None, "nearest") * Size / 32

        corr_12 = corrs_12[-1]
        corr_12 = corr_12.view(B, 16, 16, -1).permute(0, 3, 1, 2)
        corr_12 = F.interpolate(corr_12, 32, None, "bilinear", True)
        grid_x, grid_y = self.soft_argmax(corr_12)
        flow_12 = torch.cat((grid_x, grid_y), dim=1)
        flow_12 = unnormalise_and_convert_mapping_to_flow(flow_12)
        flow_up_12 = F.interpolate(flow_12, Size, None, "nearest") * Size / 32

        corr_21 = corrs_21[-1]
        corr_21 = corr_21.view(B, -1, 16, 16)
        corr_21 = F.interpolate(corr_21, 32, None, "bilinear", True)
        corr_21 = corr_21.view(B, 1024, 1024)
        corr_21 = corr_21.view(B, 32, 32, -1).permute(0, 3, 1, 2)
        grid_x, grid_y = self.soft_argmax(corr_21)
        flow_21 = torch.cat((grid_x, grid_y), dim=1)
        flow_21 = unnormalise_and_convert_mapping_to_flow(flow_21)
        flow_up_21 = F.interpolate(flow_21, Size, None, "nearest") * Size / 32

        corr_22 = corrs_22[-1]
        grid_x, grid_y = self.soft_argmax(corr_22.view(B, 32, 32, -1).permute(0, 3, 1, 2))
        flow_22 = torch.cat((grid_x, grid_y), dim=1)
        flow_22 = unnormalise_and_convert_mapping_to_flow(flow_22)
        flow_up_22 = F.interpolate(flow_22, Size, None, "nearest") * Size / 32

        x_t = torch.cat([flow_up_11, refine_feat, flow_up_12, flow_up_21, flow_up_22], dim=1)
        flow_refine = self.SRBranches(x_t)
        curr_flow = flow_up_22 + flow_refine
        return curr_flow, corr_22, mask_indexs, src_feat


if __name__ == "__main__":
    trg = torch.rand((3, 3, 512, 512))
    src = torch.rand((3, 3, 512, 512))
    b, c, h, w = trg.shape
    print(b, c, h, w)
    model = LPMFlow(num_heads=8)
    total = sum([param.nelement() for param in model.parameters()])
    print("######## Number of parameter: %.2fM ########" % (total / 1e6))
    flow, corr, mask_indexs, src_feat = model(trg, src, True)

    print(flow.shape, corr.shape)
    print(mask_indexs)
    for it in mask_indexs:
        print(it.shape)
