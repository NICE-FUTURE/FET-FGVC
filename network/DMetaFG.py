""" modified based on MetaFormer: https://github.com/dqshuai/MetaFormer. The core modification is marked with a comment beginning with NOTE
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
from modules.metaformer_MBConv import MBConvBlock
from modules.metaformer_MHSA import MHSABlock, AdaMHSABlock, Mlp


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "MetaFG_0": _cfg(),
    "MetaFG_1": _cfg(),
    "MetaFG_2": _cfg(),
}


def make_blocks(
    stage_index,
    depths,
    embed_dims,
    img_size,
    dpr,
    extra_token_num=1,
    num_heads=8,
    mlp_ratio=4.0,
    stage_type="conv",
    pruning_loc=[],
):
    blocks = []
    for block_idx in range(depths[stage_index]):
        stride = 2 if block_idx == 0 and stage_index != 1 else 1
        in_chans = (
            embed_dims[stage_index] if block_idx != 0 else embed_dims[stage_index - 1]
        )
        out_chans = embed_dims[stage_index]
        image_size = img_size if block_idx == 0 or stage_index == 1 else img_size // 2
        drop_path_rate = dpr[sum(depths[1:stage_index]) + block_idx]
        if stage_type == "conv":
            blocks.append(
                MBConvBlock(
                    ksize=3,
                    input_filters=in_chans,
                    output_filters=out_chans,
                    image_size=image_size,
                    expand_ratio=int(mlp_ratio),
                    stride=stride,
                    drop_connect_rate=drop_path_rate,
                )
            )
        elif stage_type == "mhsa":
            if len(pruning_loc) == 0 or block_idx < pruning_loc[0]:
                blocks.append(
                    MHSABlock(
                        input_dim=in_chans,
                        output_dim=out_chans,
                        image_size=image_size,
                        stride=stride,
                        num_heads=num_heads,
                        extra_token_num=extra_token_num,
                        mlp_ratio=mlp_ratio,
                        drop_path=drop_path_rate,
                    )
                )
            else:
                # NOTE
                blocks.append(
                    AdaMHSABlock(
                        input_dim=in_chans,
                        output_dim=out_chans,
                        image_size=image_size,
                        stride=stride,
                        num_heads=num_heads,
                        extra_token_num=extra_token_num,
                        mlp_ratio=mlp_ratio,
                        drop_path=drop_path_rate,
                    )
                )
        else:
            raise NotImplementedError("We only support conv and mhsa")
    return blocks


class PredictorLG(nn.Module):
    """ Importance Score Predictor
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, input_x, mask=None, ratio=0.5, ignore_token=False):
        if self.training and mask is not None:
            x1, x2 = input_x
            input_x = x1 * mask + x2 * (1 - mask)
        else:
            x1 = input_x
            x2 = input_x
            
        x = self.in_conv(input_x)
        B, N, C = x.size()
        local_x = x[:, :, :C//2]
        global_x = torch.mean(x[:, :, C//2:], keepdim=True, dim=(1))
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=2)
        pred_score = self.out_conv(x)

        if self.training:
            mask = F.gumbel_softmax(pred_score, hard=True, dim=2)[:, :, 0:1]
            if ignore_token:
                mask = mask[:, 1:]  # NOTE
            return [x1, x2], mask
        else:
            score = pred_score[:, : , 0]
            if ignore_token:
                score = score[:, 1:]  # NOTE
            B, N = score.shape
            num_keep_node = int(N * ratio)
            idx = torch.argsort(score, dim=1, descending=True)
            idx1 = idx[:, :num_keep_node]
            idx2 = idx[:, num_keep_node:]
            return input_x, [idx1, idx2]


class AdaMetaFG(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        conv_embed_dims=[64, 96, 192],
        attn_embed_dims=[384, 768],
        conv_depths=[2, 2, 3],
        attn_depths=[5, 2],
        num_heads=32,
        extra_token_num=1,
        mlp_ratio=4.0,
        conv_norm_layer=nn.BatchNorm2d,
        attn_norm_layer=nn.LayerNorm,
        conv_act_layer=nn.ReLU,
        drop_path_rate=0.0,
        only_last_cls=False,
        pruning_loc=[],
        sparse_ratio=[],
    ):
        super().__init__()
        # NOTE
        assert len(sparse_ratio) == len(
            pruning_loc
        ), "len(sparse_ratio) != len(pruning_loc)"
        assert (
            extra_token_num == 1
        ), "Ada version only supports extra_token_num==1 currently"
        self.sparse_ratio = sparse_ratio
        self.pruning_locs = pruning_loc

        self.only_last_cls = only_last_cls
        self.img_size = img_size
        self.num_classes = num_classes
        stem_chs = (3 * (conv_embed_dims[0] // 4), conv_embed_dims[0])
        dpr = [
            x.item()
            for x in torch.linspace(
                0, drop_path_rate, sum(conv_depths[1:] + attn_depths)
            )
        ]
        # stage_0 conv
        self.stage_0 = nn.Sequential(
            *[
                nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                conv_norm_layer(stem_chs[0]),
                conv_act_layer(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                conv_norm_layer(stem_chs[1]),
                conv_act_layer(inplace=True),
                nn.Conv2d(
                    stem_chs[1], conv_embed_dims[0], 3, stride=1, padding=1, bias=False
                ),
            ]
        )
        self.bn1 = conv_norm_layer(conv_embed_dims[0])
        self.act1 = conv_act_layer(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # stage_1 conv
        self.stage_1 = nn.ModuleList(
            make_blocks(
                1,
                conv_depths + attn_depths,
                conv_embed_dims + attn_embed_dims,
                img_size // 4,
                dpr=dpr,
                num_heads=num_heads,
                extra_token_num=extra_token_num,
                mlp_ratio=mlp_ratio,
                stage_type="conv",
            )
        )
        # stage_2 conv
        self.stage_2 = nn.ModuleList(
            make_blocks(
                2,
                conv_depths + attn_depths,
                conv_embed_dims + attn_embed_dims,
                img_size // 4,
                dpr=dpr,
                num_heads=num_heads,
                extra_token_num=extra_token_num,
                mlp_ratio=mlp_ratio,
                stage_type="conv",
            )
        )

        # stage_3 attn # NOTE
        self.cls_token_1 = nn.Parameter(torch.zeros(1, 1, attn_embed_dims[0]))
        self.stage_3 = nn.ModuleList(
            make_blocks(
                3,
                conv_depths + attn_depths,
                conv_embed_dims + attn_embed_dims,
                img_size // 8,
                dpr=dpr,
                num_heads=num_heads,
                extra_token_num=extra_token_num,
                mlp_ratio=mlp_ratio,
                stage_type="mhsa",
                pruning_loc=pruning_loc,
            )
        )
        # NOTE
        if len(sparse_ratio) > 0:
            predictor_list = [
                PredictorLG(attn_embed_dims[0]) for _ in range(len(pruning_loc))
            ]
            self.score_predictor = nn.ModuleList(predictor_list)

        # stage_4 attn
        self.cls_token_2 = nn.Parameter(torch.zeros(1, 1, attn_embed_dims[1]))
        self.stage_4 = nn.ModuleList(
            make_blocks(
                4,
                conv_depths + attn_depths,
                conv_embed_dims + attn_embed_dims,
                img_size // 16,
                dpr=dpr,
                num_heads=num_heads,
                extra_token_num=extra_token_num,
                mlp_ratio=mlp_ratio,
                stage_type="mhsa",
            )
        )
        self.norm_2 = attn_norm_layer(attn_embed_dims[1])
        # Aggregate
        if not self.only_last_cls:
            self.cl_1_fc = nn.Sequential(
                *[
                    Mlp(
                        in_features=attn_embed_dims[0], out_features=attn_embed_dims[1]
                    ),
                    attn_norm_layer(attn_embed_dims[1]),
                ]
            )
            self.aggregate = torch.nn.Conv1d(
                in_channels=2, out_channels=1, kernel_size=1
            )
            self.norm_1 = attn_norm_layer(attn_embed_dims[0])
            self.norm = attn_norm_layer(attn_embed_dims[1])

        self.num_features = attn_embed_dims[-1]
        patch_size = self.stage_4[0].patch_embed.patch_size
        self.num_patches = (img_size // patch_size[0]) * (img_size // patch_size[1])

        # Classifier head
        self.head = (
            nn.Linear(attn_embed_dims[-1], num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        trunc_normal_(self.cls_token_1, std=0.02)
        trunc_normal_(self.cls_token_2, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token_1", "cls_token_2"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        # stage 0 conv
        extra_tokens_1 = [self.cls_token_1]
        extra_tokens_2 = [self.cls_token_2]
        B = x.shape[0]
        x = self.stage_0(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        # stage 1 conv
        for blk in self.stage_1:
            x = blk(x)

        # stage 2 conv
        for blk in self.stage_2:
            x = blk(x)

        feature_list = []
        attn_list = []  # NOTE

        # stage 3 attn  # NOTE use dynamic designs here
        H0, W0 = self.img_size // 8, self.img_size // 8
        pruning_loc = 0
        mask = None
        decision_mask_list = []
        final_attn = None
        if len(self.sparse_ratio) == 0:
            for ind, blk in enumerate(self.stage_3):
                if ind == 0:
                    x, v, attn = blk(x, extra_tokens_1)
                else:
                    x, v, attn = blk(x)
                attn = attn[:, :, 0, 1:]
                final_attn = attn if final_attn is None else attn + final_attn
        else:
            for ind, blk in enumerate(self.stage_3):
                if ind == 0:
                    x, v, attn = blk(x, extra_tokens_1)
                else:  # NOTE mask predictor won't be placed before the first block
                    if ind in self.pruning_locs:
                        x, mask = self.score_predictor[pruning_loc](
                            x, mask, self.sparse_ratio[pruning_loc], ignore_token=True
                        )
                        pruning_loc += 1
                        decision_mask_list.append(mask)
                    if ind < self.pruning_locs[0]:
                        x, v, attn = blk(x)
                    else:
                        x, v, attn = blk(x, mask=mask)
                attn = attn[:, :, 0, 1:]
                final_attn = attn if final_attn is None else attn + final_attn
            if self.training:
                assert mask is not None
                mask = torch.cat(
                    [
                        torch.ones(
                            (mask.shape[0], 1, mask.shape[2]),
                            dtype=mask.dtype,
                            device=mask.device,
                        ),
                        mask,
                    ],
                    dim=1,
                )
                x = x[0] * mask + x[1] * (1 - mask)
        feature_list.append(x[:, 0:1] + x[:, 1:])
        # NOTE
        N = int(math.sqrt(final_attn.shape[2]))
        final_attn = torch.mean(final_attn, dim=1)
        final_attn = final_attn.view(B, N, N)
        attn_list.append(final_attn)
        L = N**2

        # stage 4 attn
        if not self.only_last_cls:
            cls_1 = x[:, :1, :]
            cls_1 = self.norm_1(cls_1)
            cls_1 = self.cl_1_fc(cls_1)
        x = x[:, 1:, :]
        H1, W1 = self.img_size // 16, self.img_size // 16
        x = x.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        final_attn = None
        for ind, blk in enumerate(self.stage_4):
            if ind == 0:
                x, v, attn = blk(x, extra_tokens_2)
            else:
                x, v, attn = blk(x)
            attn = attn[:, :, 0, 1:]
            final_attn = attn if final_attn is None else attn + final_attn
        feature_list.append(x[:, 0:1] + x[:, 1:])
        # NOTE
        N = int(math.sqrt(final_attn.shape[2]))
        final_attn = torch.mean(final_attn, dim=1)
        final_attn = final_attn.view(B, N, N)
        attn_list.append(final_attn)

        # NOTE fuse attention maps
        target_shape = (attn_list[-1].shape[1], attn_list[-1].shape[2])
        final_attn = None
        weights = np.array([i for i in range(len(attn_list), 0, -1)])
        weights = weights / weights.sum()
        for idx, attn in enumerate(attn_list):
            attn = F.interpolate(attn.unsqueeze(1), target_shape).squeeze(1)
            final_attn = (
                weights[idx] * attn
                if final_attn is None
                else (weights[idx] * attn + final_attn)
            )
        final_attn = final_attn.reshape((B,-1,1))  # B,N,N --> B,N,1

        ### NOTE process decision mask
        decision_mask = torch.ones(B, L, requires_grad=False, device=x.device)
        if self.training:
            for item in decision_mask_list:
                decision_mask *= item.squeeze(-1)
        else:
            for long_mask, short_mask in decision_mask_list:
                temp = torch.ones(B, L, device=x.device)
                offset = (
                    torch.arange(B, dtype=torch.long, device=x.device).reshape(B, 1) * L
                )
                short_mask = short_mask + offset
                temp.view(B * L)[short_mask.reshape(-1)] = 0
                decision_mask *= temp

        # output
        cls_2 = x[:, :1, :]
        cls_2 = self.norm_2(cls_2)
        if not self.only_last_cls:
            cls = torch.cat((cls_1, cls_2), dim=1)  # B,2,C
            cls = self.aggregate(cls).squeeze(dim=1)  # B,C
            cls = self.norm(cls)
        else:
            cls = cls_2.squeeze(dim=1)

        token_x = cls.unsqueeze(dim=1) + x[:, 1:]
        cls = token_x.mean(dim=1)

        # NOTE
        return (
            cls,
            token_x,
            final_attn,
            decision_mask.unsqueeze(-1).detach(),
            decision_mask_list,
            feature_list,
        )

    def forward(self, x, meta=None):
        raise NotImplementedError


class MetaFG_Teacher(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        conv_embed_dims=[64, 96, 192],
        attn_embed_dims=[384, 768],
        conv_depths=[2, 2, 3],
        attn_depths=[5, 2],
        num_heads=32,
        extra_token_num=1,
        mlp_ratio=4.0,
        conv_norm_layer=nn.BatchNorm2d,
        attn_norm_layer=nn.LayerNorm,
        conv_act_layer=nn.ReLU,
        drop_path_rate=0.0,
        only_last_cls=False,
    ):
        super().__init__()
        self.only_last_cls = only_last_cls
        self.img_size = img_size
        self.num_classes = num_classes
        stem_chs = (3 * (conv_embed_dims[0] // 4), conv_embed_dims[0])
        dpr = [
            x.item()
            for x in torch.linspace(
                0, drop_path_rate, sum(conv_depths[1:] + attn_depths)
            )
        ]
        # stage_0
        self.stage_0 = nn.Sequential(
            *[
                nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                conv_norm_layer(stem_chs[0]),
                conv_act_layer(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                conv_norm_layer(stem_chs[1]),
                conv_act_layer(inplace=True),
                nn.Conv2d(
                    stem_chs[1], conv_embed_dims[0], 3, stride=1, padding=1, bias=False
                ),
            ]
        )
        self.bn1 = conv_norm_layer(conv_embed_dims[0])
        self.act1 = conv_act_layer(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # stage_1
        self.stage_1 = nn.ModuleList(
            make_blocks(
                1,
                conv_depths + attn_depths,
                conv_embed_dims + attn_embed_dims,
                img_size // 4,
                dpr=dpr,
                num_heads=num_heads,
                extra_token_num=extra_token_num,
                mlp_ratio=mlp_ratio,
                stage_type="conv",
            )
        )
        # stage_2
        self.stage_2 = nn.ModuleList(
            make_blocks(
                2,
                conv_depths + attn_depths,
                conv_embed_dims + attn_embed_dims,
                img_size // 4,
                dpr=dpr,
                num_heads=num_heads,
                extra_token_num=extra_token_num,
                mlp_ratio=mlp_ratio,
                stage_type="conv",
            )
        )

        # stage_3
        self.cls_token_1 = nn.Parameter(torch.zeros(1, 1, attn_embed_dims[0]))
        self.stage_3 = nn.ModuleList(
            make_blocks(
                3,
                conv_depths + attn_depths,
                conv_embed_dims + attn_embed_dims,
                img_size // 8,
                dpr=dpr,
                num_heads=num_heads,
                extra_token_num=extra_token_num,
                mlp_ratio=mlp_ratio,
                stage_type="mhsa",
            )
        )

        # stage_4
        self.cls_token_2 = nn.Parameter(torch.zeros(1, 1, attn_embed_dims[1]))
        self.stage_4 = nn.ModuleList(
            make_blocks(
                4,
                conv_depths + attn_depths,
                conv_embed_dims + attn_embed_dims,
                img_size // 16,
                dpr=dpr,
                num_heads=num_heads,
                extra_token_num=extra_token_num,
                mlp_ratio=mlp_ratio,
                stage_type="mhsa",
            )
        )
        self.norm_2 = attn_norm_layer(attn_embed_dims[1])
        # Aggregate
        if not self.only_last_cls:
            self.cl_1_fc = nn.Sequential(
                *[
                    Mlp(
                        in_features=attn_embed_dims[0], out_features=attn_embed_dims[1]
                    ),
                    attn_norm_layer(attn_embed_dims[1]),
                ]
            )
            self.aggregate = torch.nn.Conv1d(
                in_channels=2, out_channels=1, kernel_size=1
            )
            self.norm_1 = attn_norm_layer(attn_embed_dims[0])
            self.norm = attn_norm_layer(attn_embed_dims[1])

        # Classifier head
        self.head = (
            nn.Linear(attn_embed_dims[-1], num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        trunc_normal_(self.cls_token_1, std=0.02)
        trunc_normal_(self.cls_token_2, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token_1", "cls_token_2"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x):
        extra_tokens_1 = [self.cls_token_1]
        extra_tokens_2 = [self.cls_token_2]
        B = x.shape[0]
        x = self.stage_0(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        for blk in self.stage_1:
            x = blk(x)

        for blk in self.stage_2:
            x = blk(x)

        H0, W0 = self.img_size // 8, self.img_size // 8
        for ind, blk in enumerate(self.stage_3):
            if ind == 0:
                x, v, attn = blk(x, extra_tokens_1)
            else:
                x, v, attn = blk(x)

        if not self.only_last_cls:
            cls_1 = x[:, :1, :]
            cls_1 = self.norm_1(cls_1)
            cls_1 = self.cl_1_fc(cls_1)
        x = x[:, 1:, :]
        H1, W1 = self.img_size // 16, self.img_size // 16
        x = x.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
        for ind, blk in enumerate(self.stage_4):
            if ind == 0:
                x, v, attn = blk(x, extra_tokens_2)
            else:
                x, v, attn = blk(x)

        cls_2 = x[:, :1, :]
        cls_2 = self.norm_2(cls_2)
        if not self.only_last_cls:
            cls = torch.cat((cls_1, cls_2), dim=1)  # B,2,C
            cls = self.aggregate(cls).squeeze(dim=1)  # B,C
            cls = self.norm(cls)
        else:
            cls = cls_2.squeeze(dim=1)

        token_x = cls.unsqueeze(dim=1) + x[:, 1:]
        cls = token_x.mean(dim=1)
        return cls

    def forward(self, x, meta=None):
        raise NotImplementedError


def build_metafg_2(img_size, pruning_loc, keep_rate):
    model = AdaMetaFG(
        conv_embed_dims=[128, 128, 256],
        attn_embed_dims=[512, 1024],
        conv_depths=[2, 2, 6],
        attn_depths=[14, 2],
        num_heads=8,
        mlp_ratio=4.0,
        num_classes=0,
        drop_path_rate=0.0,
        only_last_cls=False,
        extra_token_num=1,
        img_size=img_size,
        pruning_loc=pruning_loc,
        sparse_ratio=keep_rate,
    )
    return model


def build_metafg_2_teacher(img_size):
    model = MetaFG_Teacher(
        conv_embed_dims=[128, 128, 256],
        attn_embed_dims=[512, 1024],
        conv_depths=[2, 2, 6],
        attn_depths=[14, 2],
        num_heads=8,
        mlp_ratio=4.0,
        num_classes=0,
        drop_path_rate=0.0,
        only_last_cls=False,
        extra_token_num=1,
        img_size=img_size,
    )
    return model
