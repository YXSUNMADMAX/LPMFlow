import logging
from functools import partial
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from models.encoder.encoder.utils import combine_tokens
from models.encoder.encoder.vit import VisionTransformer
from models.encoder.layers.attn_blocks import Block_nmask

_logger = logging.getLogger(__name__)


class SingleTower(VisionTransformer):
    def __init__(self, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 act_layer=None, weight_init='',
                 ce_loc=[3, 4, 5], ce_keep_ratio=[1.0, 1.0, 1.0]):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.pos_embed = nn.Parameter(torch.zeros(1, 257, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # for patch embedding
        patch_pos_embed = self.pos_embed[:, 1:, :]
        src_patch_pos_embed = patch_pos_embed
        trg_patch_pos_embed = patch_pos_embed

        self.pos_embed_trg = nn.Parameter(trg_patch_pos_embed)
        self.pos_embed_src = nn.Parameter(src_patch_pos_embed)

        cls_pos_embed = self.pos_embed[:, 0:1, :]
        self.cls_pos_embed_src = nn.Parameter(cls_pos_embed)
        self.cls_pos_embed_trg = nn.Parameter(cls_pos_embed)

        self.trg_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.trg_segment_pos_embed = trunc_normal_(self.trg_segment_pos_embed, std=.02)
        self.src_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.src_segment_pos_embed = trunc_normal_(self.src_segment_pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        for i in range(depth):
            blocks.append(
                Block_nmask(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def forward_features(self, src_feat, trg_feat, mask_src=None, mask_trg=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False
                         ):

        # print(src_feat.shape)
        B, _, _= src_feat.shape[0], src_feat.shape[1], src_feat.shape[2]

        cls_token_src = src_feat[:, 0:1, :]
        cls_token_src = cls_token_src + self.cls_pos_embed_src
        cls_token_trg = trg_feat[:, 0:1, :]
        cls_token_trg = cls_token_trg + self.cls_pos_embed_trg

        patch_src = src_feat[:, 1:, :]
        patch_trg = trg_feat[:, 1:, :]

        patch_src += self.pos_embed_src
        patch_trg += self.pos_embed_trg

        patch_src += self.src_segment_pos_embed
        patch_trg += self.trg_segment_pos_embed

        src_tokens = torch.cat([cls_token_src, patch_src], dim=1)
        trg_tokens = torch.cat([cls_token_trg, patch_trg], dim=1)
        x = combine_tokens(src_tokens, trg_tokens, mode=self.cat_mode)
        x = self.pos_drop(x)

        attns = []

        for i, blk in enumerate(self.blocks):
            x, attn = blk(x)
            attns.append(attn)

        x = self.norm(x)
        return x, attns

    def forward(self, src_feat, trg_feat, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):

        x, attns = self.forward_features(src_feat, trg_feat)
        return x, attns
