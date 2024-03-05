import torch
import torch.nn as nn
import torch.nn.functional as F

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)
    

class ConvNextDistillDiffPruningLoss(nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, teacher_model, ratio_weight=10.0, distill_weight=0.5, keep_ratio=[0.9, 0.7, 0.5], swin_token=False):
        super().__init__()
        self.teacher_model = teacher_model
        self.keep_ratio = keep_ratio
        self.ratio_weight = ratio_weight
        self.distill_weight = distill_weight
        self.swin_token = swin_token

    def forward(self, inputs, outputs):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
        """
        token_feature, decision_mask_list = outputs

        # ratio loss
        ratio_loss = torch.tensor(0.0)
        ratio = self.keep_ratio
        for i, score in enumerate(decision_mask_list):
            if not self.swin_token:
                pos_ratio = score.mean(dim=(2,3))
            else:
                pos_ratio = score.mean(dim=1)
            ratio_loss = ratio_loss + ((pos_ratio - ratio[i]) ** 2).mean()
        if self.teacher_model is None:
            loss = self.ratio_weight * ratio_loss / len(decision_mask_list)
            return loss
        
        # distillation loss
        with torch.no_grad():
            cls_t = self.teacher_model.forward_features(inputs.to(inputs.device))
        cls_kl_loss = torch.pow(token_feature - cls_t, 2).mean()

        loss = self.ratio_weight * ratio_loss / len(decision_mask_list) + self.distill_weight * cls_kl_loss
        return loss


class DistillDiffPruningLoss_dynamic(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, teacher_model, ratio_weight=2.0, distill_weight=0.5, dynamic=False, pruning_loc=[3,6,9], keep_ratio=[0.75, 0.5, 0.25], mse_token=False):
        super().__init__()
        self.teacher_model = teacher_model
        self.pruning_loc = pruning_loc
        self.keep_ratio = keep_ratio
        self.mse_token = mse_token
        self.dynamic = dynamic
        self.ratio_weight = ratio_weight
        self.distill_weight = distill_weight

    def forward(self, inputs, outputs):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
        """
        token_feature, patch_feature, decision_mask, decision_mask_list = outputs

        # ratio loss
        ratio_loss = torch.tensor(0.0)
        ratio = self.keep_ratio
        for i, score in enumerate(decision_mask_list):
            if self.dynamic:
                pos_ratio = score.mean()
            else:
                pos_ratio = score.mean(1)
            ratio_loss = ratio_loss + ((pos_ratio - ratio[i]) ** 2).mean()
        if self.teacher_model is None:
            loss = self.ratio_weight * ratio_loss / len(self.pruning_loc)
            return loss

        # distillation loss
        with torch.no_grad():
            cls_t, token_t = self.teacher_model.forward_features(inputs.to(inputs.device))
        if self.mse_token:
            cls_kl_loss = torch.pow(token_feature - cls_t, 2).mean()
        else:
            raise NotImplementedError

        loss = self.ratio_weight * ratio_loss / len(self.pruning_loc) + self.distill_weight * cls_kl_loss
        return loss