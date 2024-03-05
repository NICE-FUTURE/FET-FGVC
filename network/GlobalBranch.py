import math
import os
import torch
from torch import nn
import torch.nn.functional as F

from network.LocalBranch import LocalBranch
from network.DViT import VisionTransformerDiffPruning, VisionTransformerTeacher
from network.DSwin import AdaSwinTransformer, SwinTransformer_Teacher
from network.DMetaFG import build_metafg_2, build_metafg_2_teacher
from modules.common import Mlp


supported_arch = ["vit-base", "swin-base", "metaformer-2"]

PRETRAIN_LIST = {
    "vit-base": "./vit/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz", 
    "swin-base": "./swin/swin_base_patch4_window7_224_22kto1k.pth", 
    "metaformer-2": "./metaformer/metafg_2_21k_224.pth",
}


def pdist(vectors):
    """
    vectors: (batch, 2048)
    distance: EuclideanDistance
    The more similar two images are, the higher the similarity is and the smaller the distance is
    """
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


def create_global_branch(arch:str, cfg:dict, only_teacher_model:bool=False):
    if arch.startswith("vit"):
        if arch == "vit-base":
            pruning_loc = [cfg["pruning_loc"]]# default [6]
            global_feature_dim = 768
            num_heads = 12
            depth = 12
            pretrained_path = PRETRAIN_LIST[arch]
        else:
            raise NotImplementedError
        
        if cfg["dynamic"] == False:
            pruning_loc = []
            cfg["keep_rate"] = []
        
        if only_teacher_model:
            teacher_model = VisionTransformerTeacher(
                img_size=cfg["image_size"], patch_size=16, num_classes=cfg["num_classes"], 
                drop_rate=cfg["drop_rate"], attn_drop_rate=cfg["attn_drop_rate"], drop_path_rate=cfg["drop_path_rate"], 
                num_heads=num_heads, embed_dim=global_feature_dim, depth=depth
            )
            if pretrained_path.endswith(".npz"):
                teacher_model.load_pretrained(pretrained_path)
            elif pretrained_path.endswith(".pth"):
                checkpoint = torch.load(pretrained_path, map_location="cpu")
                checkpoint = {k:v for k,v in checkpoint.items() if not k.startswith("head")}
                teacher_model.load_state_dict(checkpoint, strict=False)
            else:
                raise NotImplementedError
            return teacher_model
        else:
            model = VisionTransformerDiffPruning(
                img_size=cfg["image_size"], patch_size=16, num_classes=cfg["num_classes"], 
                pruning_loc=pruning_loc, token_ratio=cfg["keep_rate"], 
                drop_rate=cfg["drop_rate"], attn_drop_rate=cfg["attn_drop_rate"], drop_path_rate=cfg["drop_path_rate"], 
                num_heads=num_heads, embed_dim=global_feature_dim, depth=depth
            )
            if cfg["load_pretrained"]:
                if pretrained_path.endswith(".npz"):
                    model.load_pretrained(pretrained_path)
                elif pretrained_path.endswith(".pth"):
                    checkpoint = torch.load(pretrained_path, map_location="cpu")
                    checkpoint = {k:v for k,v in checkpoint.items() if not k.startswith("head")}
                    model.load_state_dict(checkpoint, strict=False)
                else:
                    raise NotImplementedError
            return model, pruning_loc, global_feature_dim, model.patch_embed.num_patches, depth
        
    elif arch.startswith("swin"):
        if arch == 'swin-base':
            embed_dim = 128
            depth = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            pruning_loc = [cfg["pruning_loc"]]
            pretrained_path = PRETRAIN_LIST[arch]
        else:
            raise NotImplementedError
        
        if cfg["dynamic"] == False:
            pruning_loc = []
            cfg["keep_rate"] = []

        if only_teacher_model:
            teacher_model = SwinTransformer_Teacher(
                img_size=cfg["image_size"], num_classes=0, window_size=cfg["window_size"], 
                embed_dim=embed_dim, depths=depth, num_heads=num_heads,
            )
            checkpoint = torch.load(pretrained_path, map_location="cpu")["model"]
            teacher_model.load_state_dict(checkpoint, strict=False)
            return teacher_model
        else:
            model = AdaSwinTransformer(
                img_size=cfg["image_size"], num_classes=0, window_size=cfg["window_size"], 
                embed_dim=embed_dim, depths=depth, num_heads=num_heads,
                pruning_loc=pruning_loc, sparse_ratio=cfg["keep_rate"]
            )
            global_feature_dim = model.num_features
            if cfg["load_pretrained"]:
                checkpoint = torch.load(pretrained_path, map_location="cpu")["model"]
                model.load_state_dict(checkpoint, strict=False)
            num_patches = model.layers[-1].input_resolution[0] * model.layers[-1].input_resolution[1] # type: ignore
            return model, pruning_loc, global_feature_dim, num_patches, len(depth)

    elif arch.startswith("metaformer"):
        if arch == 'metaformer-2':
            pruning_loc = [cfg["pruning_loc"]]# default [6]
            pretrained_path = PRETRAIN_LIST[arch]
        else:
            raise NotImplementedError
        
        if cfg["dynamic"] == False:
            pruning_loc = []
            cfg["keep_rate"] = []

        if only_teacher_model:
            teacher_model = build_metafg_2_teacher(
                img_size=cfg["image_size"]
            )
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            teacher_model.load_state_dict(checkpoint, strict=False)
            return teacher_model
        else:
            model = build_metafg_2(
                img_size=cfg["image_size"], 
                pruning_loc=pruning_loc, 
                keep_rate=cfg["keep_rate"]
            )
            if cfg["load_pretrained"]:
                checkpoint = torch.load(pretrained_path, map_location="cpu")
                model.load_state_dict(checkpoint, strict=False)
            global_feature_dim = model.num_features
            num_patches = model.num_patches
            num_attn_stages = 2
            return model, pruning_loc, global_feature_dim, num_patches, num_attn_stages

    else:
        raise NotImplementedError


def create_local_branch(cfg:dict):
        local_branch = LocalBranch(
            depth=cfg["depth"], 
            embed_dim=cfg["embed_dim"], 
            num_parts=cfg["num_parts"], 
            part_channels=cfg["part_channels"], 
            batch_size=cfg["batch_size"], 
            num_patches=cfg["num_patches"],
            gaussian_ksize=cfg["gaussian_ksize"],
        )
        return local_branch


class FET_FGVC(nn.Module):

    def __init__(self, arch, model_cfg, local_cfg, nopfi, nolocal, pretrain_root, local_from_stage=2):
        super(FET_FGVC, self).__init__()

        global PRETRAIN_LIST        
        for k, v in PRETRAIN_LIST.items():
            PRETRAIN_LIST[k] = os.path.abspath(os.path.expanduser(os.path.join(pretrain_root, v)))

        self.nopfi = nopfi
        self.nolocal = nolocal
        if arch.startswith("vit"):
            self.has_cls_token = True
        else:
            self.has_cls_token = False

        ### global branch
        base_rate = model_cfg["base_keep_rate"]  # default 0.7
        self.keep_rate = [base_rate]

        self.backbone, self.pruning_loc, global_feature_dim, num_patches, num_stages = create_global_branch(
            arch, cfg={
            **model_cfg, 
            "keep_rate": self.keep_rate
        })  # type: ignore

        if arch.startswith("swin"):
            assert num_stages == 4
            global_feature_dim_list = [256, 512, 1024, 1024]
            if model_cfg["image_size"] == 224:
                num_patches_list = [784, 196, 49, 49]
            elif model_cfg["image_size"] == 448:
                num_patches_list = [3136, 784, 196, 196]
            else:
                raise NotImplementedError
        elif arch.startswith("metaformer"):
            assert num_stages == 2
            global_feature_dim_list = [512, 1024]
            if model_cfg["image_size"] == 224:
                num_patches_list = [196, 49]
            elif model_cfg["image_size"] == 448:
                num_patches_list = [784, 196]
            else:
                raise NotImplementedError
        else:
            global_feature_dim_list = [global_feature_dim]*num_stages
            num_patches_list = [num_patches]*num_stages

        self.local_from_stage = local_from_stage

        ### local branch
        if not self.nolocal:
            self.local_feature_dim = global_feature_dim_list[self.local_from_stage]
            self.local_branch = create_local_branch(
                cfg={
                    **local_cfg,
                    "embed_dim": self.local_feature_dim,
                    "num_patches": num_patches_list[self.local_from_stage],  # local branch
                }
            )

        ### feature interaction module
        if not self.nopfi:
            self.norm = nn.LayerNorm(global_feature_dim)
            self.avgpool = nn.AdaptiveAvgPool1d(1)

        ### classifier
        if not self.nolocal:
            self.local_proj_layer = nn.Linear(self.local_feature_dim, global_feature_dim)
            self.classifier = Mlp(global_feature_dim, global_feature_dim*4, model_cfg["num_classes"])
        else:
            self.classifier = nn.Linear(global_feature_dim, model_cfg["num_classes"])

        self.softmax_layer = nn.LogSoftmax(dim=1)

    def get_param_groups(self):
        param_names = [[], []]
        param_groups = [[], []]  # backbone, otherwise
        for name, param in super().named_parameters():
            if name.startswith("backbone") and "score_predictor" not in name:
                param_names[0].append(name)
                param_groups[0].append(param)
            else:
                param_names[1].append(name)
                param_groups[1].append(param)
        return param_names, param_groups

    def forward(self, images, targets=None, flag=None):
        ### global features
        global_features, global_patch_features, global_attentions, decision_mask, decision_mask_list, feature_list = self.backbone.forward_features(images)  # global_values is global_patch_features in practice

        ### obtain patch feature by using deep token embeddings of global branch directly
        patch_features = feature_list[self.local_from_stage].detach()  # local branch

        target_size = int(math.sqrt(patch_features.shape[1]))
        B, L, C = decision_mask.shape
        size = int(math.sqrt(L))
        decision_mask = decision_mask.permute(0,2,1).reshape(B, C, size, size) # apply reshape on feature here  # B, C0, h0, w0
        decision_mask = F.interpolate(decision_mask, (target_size, target_size))
        decision_mask = decision_mask.permute(0,2,3,1).reshape(B,target_size**2,C)

        ### local features
        parts_masks = None
        if self.nolocal:
            local_features = None
        else:
            assert patch_features.shape[2] == self.local_feature_dim, f"patch_features.shape[{self.local_from_stage}]:{patch_features.shape[2]} == self.local_feature_dim:{self.local_feature_dim}"
            local_features, parts_masks = self.local_branch(decision_mask, patch_features)  # B, D

        # combine global features with local features
        features = self.fuse_feature(global_feature=global_features, local_feature=local_features)
        logits = self.classifier(features)

        if flag == "visual":
            if self.nolocal:
                return features, patch_features, decision_mask, None
            else:
                _, parts_masks = self.local_branch.locate_parts(decision_mask, patch_features)
                B, D, H, W = parts_masks.shape  # D: num_parts
                parts_masks = parts_masks.permute(0,2,3,1).reshape(B, -1, D)
                return features, patch_features, decision_mask, parts_masks, global_attentions
        
        ### global feature interaction
        if self.training and not self.nopfi:
            with torch.no_grad():
                intra_pairs, inter_pairs, intra_labels, inter_labels = self.get_pairs(global_features, targets)

            features1_self, features1_other, features2_self, features2_other = self.attention_pfi(global_features, global_patch_features, global_attentions, intra_pairs, inter_pairs)

            # obtain classification probability
            logit1_self =  self.classifier(features1_self)
            logit1_other = self.classifier(features1_other)
            logit2_self =  self.classifier(features2_self)
            logit2_other = self.classifier(features2_other)

            # prepare cross-entropy calculation
            labels1 = torch.cat([intra_labels[:, 0], inter_labels[:, 0]], dim=0)  # ori samples and ori samples
            labels2 = torch.cat([intra_labels[:, 1], inter_labels[:, 1]], dim=0)  # intra and inter samples
            self_logits = torch.cat([logit1_self, logit2_self], dim=0)
            other_logits = torch.cat([logit1_other, logit2_other], dim=0)
            ce_logits = torch.cat([self_logits, other_logits], dim=0)  # 8B, num_classes
            ce_targets = torch.cat([labels1, labels2, labels1, labels2], dim=0)  # 8B, num_classes

            # prepare margin rank loss calculation
            self_scores = self.softmax_layer(self_logits)
            self_scores = self_scores[torch.arange(self_scores.shape[0]), torch.cat([labels1, labels2], dim=0).to(torch.long)]
            other_scores = self.softmax_layer(other_logits)
            other_scores = other_scores[torch.arange(other_scores.shape[0]), torch.cat([labels1, labels2], dim=0).to(torch.long)]

            if self.has_cls_token:
                return logits, ce_logits, ce_targets, self_scores, other_scores, global_features, decision_mask_list, parts_masks, global_patch_features, decision_mask
            else:
                return logits, ce_logits, ce_targets, self_scores, other_scores, global_features, decision_mask_list, parts_masks
        else:
            # without feature interaction
            if self.training and self.nopfi:
                if self.has_cls_token:
                    return logits, global_features, decision_mask_list, parts_masks, global_patch_features, decision_mask
                else:
                    return logits, global_features, decision_mask_list, parts_masks
            else:
                # validating, with or without pfi
                return logits

    def _cal_attention_features(self, v, attn):
        x = v * attn
        x = self.norm(x)  # B,N,D
        x = self.avgpool(x.transpose(1, 2))  # B,D,N --> B,D,1
        x = torch.flatten(x, 1)  # B,D
        return x

    def attention_pfi(self, features, values, attentions, intra_pairs, inter_pairs):
        features1 = torch.cat([features[intra_pairs[:, 0]], features[inter_pairs[:, 0]]], dim=0)
        features2 = torch.cat([features[intra_pairs[:, 1]], features[inter_pairs[:, 1]]], dim=0)
        values1 = torch.cat([values[intra_pairs[:, 0]], values[inter_pairs[:, 0]]], dim=0)
        values2 = torch.cat([values[intra_pairs[:, 1]], values[inter_pairs[:, 1]]], dim=0)

        attentions1 = torch.cat([attentions[intra_pairs[:, 0]], attentions[inter_pairs[:, 0]]], dim=0)
        attentions2 = torch.cat([attentions[intra_pairs[:, 1]], attentions[inter_pairs[:, 1]]], dim=0)

        features1_self = self._cal_attention_features(values1, attentions1) + features1
        features1_other = self._cal_attention_features(values1, attentions2) + features1
        features2_self = self._cal_attention_features(values2, attentions2) + features2
        features2_other = self._cal_attention_features(values2, attentions2) + features2
        return features1_self, features1_other, features2_self, features2_other

    def get_pairs(self, embeddings, labels):
        distance_matrix = pdist(embeddings)  # Calculate the similarity between images in a batch

        labels = labels.unsqueeze(dim=1)
        batch_size = embeddings.shape[0]
        lb_eqs = (labels == torch.t(labels))

        dist_same = distance_matrix.clone()
        lb_eqs = lb_eqs.fill_diagonal_(fill_value=False, wrap=False)  # Positions with the same category are True; Positions with different categories are False; Positions with themselves are False
        dist_same[lb_eqs == False] = float("inf")  # Samples of different classes are infinitely far away; The distance between self and self is infinite. There is an effective distance only between samples with the same class
        intra_idxs = torch.argmin(dist_same, dim=1)  # intra; Match similar images within a class

        dist_diff = distance_matrix.clone()
        lb_eqs = lb_eqs.fill_diagonal_(fill_value=True, wrap=False)   # Same category is True. Different classes are False. Self and its position are True
        dist_diff[lb_eqs == True] = float("inf")  # The distance between samples with the same class is infinite. The distance between oneself and oneself is infinite; There is only an effective distance between samples with different classes
        inter_idxs = torch.argmin(dist_diff, dim=1)  # inter; Match images that are similar between classes

        intra_labels = torch.cat([labels[:], labels[intra_idxs]], dim=1)
        inter_labels = torch.cat([labels[:], labels[inter_idxs]], dim=1)
        intra_pairs = torch.cat([torch.arange(0,batch_size).unsqueeze(dim=1).to(embeddings.device), intra_idxs.unsqueeze(dim=1)], dim=1)
        inter_pairs  = torch.cat([torch.arange(0,batch_size).unsqueeze(dim=1).to(embeddings.device), inter_idxs.unsqueeze(dim=1)], dim=1)

        # intra_pairs, inter_pairs
        # intra_labels, inter_labels
        # pairs: Original sample index - Corresponding similar sample index
        return intra_pairs, inter_pairs, intra_labels, inter_labels

    def fuse_feature(self, global_feature, local_feature=None):
        """ fuse global feature with local feature
        global feature: produced by backbone or PFI module
        local feature: produced by local branch
        """
        if local_feature is not None:
            ### addition
            local_feature = self.local_proj_layer(local_feature)
            feature = global_feature + 0.5 * local_feature
            return feature
        else:
            return global_feature
