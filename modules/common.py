import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        B, xN, C = x.shape
        _, cN, _ = context.shape
        q = self.q(x).reshape(B, xN, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B,H,N,D
        kv = self.kv(context).reshape(B, cN, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 2, B,H,N,D
        k, v = kv[0], kv[1]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, xN, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class UpsampleFeature(nn.Module):
    def __init__(self, feature_dim, feature_dims, hidden_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.feature_dims = feature_dims
        assert len(self.feature_dims) == 4, "only support 4 groups of features"

        self.fc_list = nn.ModuleList()
        self.flag_list = []
        for dim in self.feature_dims:
            if dim != self.hidden_dim:
                self.fc_list.append(nn.Conv2d(dim, self.hidden_dim, kernel_size=1, bias=False))
                self.flag_list.append(1)
            else:
                self.flag_list.append(0)

        self.fuse_fc = nn.Sequential(
            nn.Conv2d(self.hidden_dim*4, self.feature_dim, kernel_size=1, bias=False), 
            nn.BatchNorm2d(self.feature_dim), 
            nn.ReLU()
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, feature_list):
        target_size = int(math.sqrt(feature_list[0].shape[1]))
        fc_idx = 0
        new_feature_list = []
        for feat_idx, flag in enumerate(self.flag_list):
            feature = feature_list[feat_idx]
            B, L, C = feature.shape
            size = int(math.sqrt(L))
            feature = feature.permute(0,2,1).view(B, C, size, size) # apply reshape on feature here  # B, C0, h0, w0
            if flag == 1:
                feature = self.fc_list[fc_idx](feature)  # B, C, h0, w0
                # apply upsample on feature here  # B, C, h, w
                fc_idx += 1
            feature = F.interpolate(feature, (target_size, target_size), mode="bilinear", align_corners=False)
            new_feature_list.append(feature)
        
        feature = torch.cat(new_feature_list, dim=1)  # B, 4C, h, w
        feature = self.fuse_fc(feature)  # B, C, h, w
        feature = self.dropout(feature)
        # apply reshape on feature here  # B, L1, C
        feature = feature.view(B,C,target_size**2).permute(0,2,1)
        
        return feature  # B, L1, C
