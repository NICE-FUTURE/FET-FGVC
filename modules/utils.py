# -*- "coding: utf-8" -*-

import torch
import torch.backends.cudnn
import torch.nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.modules.batchnorm import _BatchNorm
import math
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import random


def fix_seed(seed=2):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cal_eta(start_time, cur, total):
    time_now = datetime.now()
    time_now = time_now.replace(microsecond=0)
    scale = (total-cur) / float(cur)
    delta = (time_now - start_time)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(eta)


def plot_history(loss_list, acc_list, val_loss_list, val_acc_list, history_save_path):
    """
    plt loss and acc curve
    """
    plt.figure(figsize=(10,8))
    plt.subplot(211)
    plt.plot(loss_list, color="blue", label="loss")
    plt.plot(val_loss_list, color="orange", label="val_loss")
    plt.legend()
    plt.subplot(212)
    plt.axhline(y=1, color="black", linestyle="--")
    plt.axhline(y=0.8, color="black", linestyle="--")
    plt.axhline(y=0.6, color="black", linestyle="--")
    plt.plot(acc_list, color="blue", label="acc")
    plt.plot(val_acc_list, color="orange", label="val_acc")
    plt.legend()

    plt.savefig(history_save_path)
    plt.close()


def cal_accuracy(outputs, targets, k=1):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    if len(outputs.shape) == 1:
        outputs = torch.unsqueeze(outputs, dim=0)
    batch_size = targets.size(0)
    _, ind = outputs.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor

    return correct_total.item() / batch_size


# from https://github.com/TACJu/TransFG/blob/master/utils/scheduler.py
class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


def batch_index_mask(mask, idx):
    B, N = mask.shape[:2]
    cur_mask = torch.zeros_like(mask).reshape(B*N)
    offset = torch.arange(B, dtype=torch.long, device=mask.device).view(B, 1) * N
    idx = idx + offset
    cur_mask[idx.reshape(-1)] = 1
    mask = cur_mask.reshape(B, N, 1) * mask
    return mask


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


def batch_index_fill(x, x1, x2, idx1, idx2):
    B, N, C = x.size()
    B, N1, C = x1.size()
    B, N2, C = x2.size()

    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1)
    idx1 = idx1 + offset * N
    idx2 = idx2 + offset * N

    x = x.reshape(B*N, C)

    x[idx1.reshape(-1)] = x1.reshape(B*N1, C)
    x[idx2.reshape(-1)] = x2.reshape(B*N2, C)

    x = x.reshape(B, N, C)
    return x


def generate_gaussian_kernel(l=5, sig=1.):
    """ https://stackoverflow.com/a/43346070/11285283
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


@torch.no_grad()        
def smooth_decision_mask(kernel, kernel_size, decision_mask):
    B, L, C = decision_mask.shape
    size = int(math.sqrt(L))
    decision_mask = decision_mask.permute(0,2,1).reshape(B, C, size, size)  # apply reshape on feature here  # B, C0, h0, w0
    decision_mask = F.conv2d(decision_mask, kernel, padding=kernel_size//2)  # (batch_size,in_channel,num_tokens)
    decision_mask = decision_mask.permute(0,2,3,1).reshape(B,L,C)  # B1L -> BL1
    return decision_mask