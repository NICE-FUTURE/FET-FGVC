# -*- "coding: utf-8" -*-

from datetime import datetime
import math
import time
import os
import logging
import traceback
import argparse
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import numpy as np
import matplotlib
matplotlib.use("agg")

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from timm.layers import resample_abs_pos_embed

from thop import profile
from network.GlobalBranch import FET_FGVC, create_global_branch, supported_arch
from modules.datasets import BatchDataset, BalancedBatchSampler
from modules import utils, losses


utils.fix_seed()


def main():
    # prepare model
    load_pretrained = False if (args.from_scratch or args.finetune or args.vis_mode) else True
    if args.img_size == 224:
        size = 224
    elif args.img_size == 448:
        size = 384
    batch_size = (args.sample_classes*args.sample_images)
    model_cfg = {
        "num_classes": cfg.dataset.num_classes, 
        "image_size": args.img_size, 
        "size": size, 
        "load_pretrained": load_pretrained, 
        "drop_rate": cfg.backbone.drop_rate, 
        "attn_drop_rate": cfg.backbone.drop_path_rate, 
        "drop_path_rate": cfg.backbone.attn_drop_rate, 
        "window_size": args.window_size, 
        "dynamic": not args.nodynamic, 
        "base_keep_rate": args.base_keep_rate, 
        "pruning_loc": args.pruning_loc,
    }
    local_cfg = {
        "depth": args.local_depth,
        "num_parts": args.num_parts,
        "batch_size": batch_size, 
        "part_channels": args.part_channels, 
        "gaussian_ksize": args.gaussian_ksize,
    }
    model = FET_FGVC(args.arch, model_cfg, local_cfg, args.nopfi, args.nolocal, cfg.backbone.pretrain_root, args.local_from_stage)
    
    logging.info("Calculate MACs & FLOPs ...")
    inputs = torch.randn((1, 3, args.img_size, args.img_size))
    macs, num_params = profile(model, (inputs,), verbose=False) # type: ignore
    logging.info("\nParams(M):{:.2f}, MACs(G):{:.2f}, FLOPs(G):~{:.2f}".format(num_params/(1000**2), macs/(1000**3), 2*macs/(1000**3)))
    logging.info("")
    
    if args.distill:
        teacher_model:nn.Module = create_global_branch(args.arch, model_cfg, only_teacher_model=True)  # type: ignore
        teacher_model.to(device)
        teacher_model.eval()
    else:
        teacher_model = None
    logging.info("\nargs: {}".format(args))
    logging.info("\nconfigs: {}".format(cfg))
    if not args.vis_mode:
        logging.info("\nNetwork config: \n{}".format(model))

    # load trained weights
    if args.weights_dir:
        if args.finetune:
            weights_path = os.path.join(args.weights_dir, "best.pth")
        else:
            weights_path = os.path.join(args.weights_dir, "last.pth")
        logging.info("Load weights from {}".format(weights_path))
        state_dict = torch.load(weights_path, map_location="cpu")
        state_dict = {k.replace("_orig_mod.", ""):v for k,v in state_dict.items()}  # the weights of torch compiled model has a prefix '_orig_mod', can be removed
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k.endswith("attn_mask") and args.arch.startswith("swin"):
                continue  # remove attn_mask
            elif k.endswith("edge_index"):
                continue  # remove edge_index
            elif k.startswith("local_branch.before_gcn_fc"):
                continue  # remove before_gcn_fc
            elif k.endswith("relative_position_index") or k.endswith("relative_position_bias_table"):
                continue  # for metaformer fine-tuning
            elif k.endswith("edge_weight"):
                filtered_state_dict[k] = v[:args.num_parts*(args.num_parts-1)*batch_size]
            elif k.endswith("pos_embed"):
                filtered_state_dict[k] = resample_abs_pos_embed(
                    v,
                    new_size=model.backbone.patch_embed.grid_size,
                    num_prefix_tokens=1,
                    interpolation='bilinear',
                    antialias=False,
                    verbose=True,
                )
            else:
                filtered_state_dict[k] = v
        model.load_state_dict(filtered_state_dict, strict=False)
    model.to(device)

    scale_size = int(round(512*args.img_size/448))
    transform1 = transforms.Compose([
                                transforms.Resize([scale_size,scale_size]),
                                transforms.RandomCrop([args.img_size,args.img_size]),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=(0.485, 0.456, 0.406),
                                    std=(0.229, 0.224, 0.225)
                                )])
    transform2 = transforms.Compose([
                                transforms.Resize([scale_size,scale_size]),
                                transforms.CenterCrop([args.img_size,args.img_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=(0.485, 0.456, 0.406),
                                    std=(0.229, 0.224, 0.225)
                                )])
    
    [backbone_names, other_names], [backbone_params, other_params] = model.get_param_groups()
    logging.info(f"\nbackbone_names:{backbone_names}\nother_names:{other_names}\n")
    
    if not args.vis_mode:
        ### optimizers, loss functions
        if args.finetune:
            optimizers = [
                torch.optim.AdamW(model.parameters(), lr=cfg.train.backbone_lr, weight_decay=cfg.train.weight_decay, betas=cfg.train.betas),
            ]
        else:
            optimizers = [
                torch.optim.AdamW(backbone_params, lr=cfg.train.backbone_lr, weight_decay=cfg.train.weight_decay, betas=cfg.train.betas),
                torch.optim.AdamW(other_params, lr=cfg.train.others_lr, weight_decay=cfg.train.weight_decay, betas=cfg.train.betas),
            ]
        schedulers = [
            utils.WarmupCosineSchedule(optimizer, warmup_steps=cfg.train.warmup_epochs, t_total=int(1.1*args.epochs))
            for optimizer in optimizers
        ]
        ce_criterion = losses.LabelSmoothingCrossEntropy().to(device)
        mse_criterion = nn.MSELoss().to(device)
        rank_criterion = None if args.nopfi else nn.MarginRankingLoss(margin=0.05).to(device)
        if args.nodynamic:
            dynamic_criterion = None
        else:
            if model.has_cls_token:
                if args.ratio_weight is None:
                    args.ratio_weight = 2
                dynamic_criterion = losses.DistillDiffPruningLoss_dynamic(teacher_model, ratio_weight=args.ratio_weight, distill_weight=0.5, pruning_loc=model.pruning_loc, keep_ratio=model.keep_rate, mse_token=True)
            else:
                if args.ratio_weight is None:
                    args.ratio_weight = 10
                dynamic_criterion = losses.ConvNextDistillDiffPruningLoss(teacher_model, ratio_weight=args.ratio_weight, distill_weight=0.5, keep_ratio=model.keep_rate, swin_token=True)
            
        ### resume training
        start_epoch = 0
        best_val = None
        if args.weights_dir and not args.finetune:
            state_dict = torch.load(os.path.join(args.weights_dir, "params.pth"), map_location="cpu")
            start_epoch = state_dict["epoch"]
            [optimizers[idx].load_state_dict(dict) for idx,dict in enumerate(state_dict['optimizer_state_dicts'])]
            best_val = state_dict["best_val"]

        # Data loading code
        train_dataset = BatchDataset(cfg.dataset.root_dir, cfg.train.stage, cfg.dataset.txt_dir, transform=transform1)
        if args.nopfi:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cfg.train.num_workers, pin_memory=True)
        else:
            train_sampler = BalancedBatchSampler(train_dataset, args.sample_classes, args.sample_images)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=cfg.train.num_workers, pin_memory=True)

    val_dataset = BatchDataset(cfg.dataset.root_dir, cfg.val.stage, cfg.dataset.txt_dir, transform=transform2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.val.num_workers, pin_memory=True)

    if not args.vis_mode:
        logging.info('START TIME:{}'.format(time.asctime(time.localtime(time.time()))))
        start_time = datetime.now().replace(microsecond=0)
        loss_list = []
        acc_list = []
        val_loss_list = []
        val_acc_list = []
        scaler = None
        for epoch in range(start_epoch, args.epochs):
            loss, acc = train(scaler, train_loader, model, [ce_criterion, rank_criterion, dynamic_criterion, mse_criterion], optimizers, epoch)
            [scheduler.step() for scheduler in schedulers]
            loss_list.append(loss)
            acc_list.append(acc)
            # validate
            val_loss, val_acc = validate(val_loader, model, ce_criterion, epoch)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            eta = utils.cal_eta(start_time, epoch+1, args.epochs)
            logging.info(
                "[Epoch:{}/{}] eta:{} lr:{:.6f} loss:{:.6f} acc:{:.6f} val_loss:{:.6f} val_acc:{:.6f}".format(
                    epoch+1, args.epochs, eta, optimizers[0].param_groups[0]['lr'], loss, acc, val_loss, val_acc
                )
            )
            utils.plot_history(loss_list, acc_list, val_loss_list, val_acc_list, history_save_path)
            # save model
            torch.save({
                'epoch': epoch+1,
                'optimizer_state_dicts': [optimizer.state_dict() for optimizer in optimizers], 
                'best_val': best_val,
                }, params_save_path)
            torch.save(model.state_dict(), os.path.join(model_last_path))
            if best_val is None or val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(), model_best_path)
                logging.info("Saved best model.")

        utils.plot_history(loss_list, acc_list, val_loss_list, val_acc_list, history_save_path)
    logging.info('STOP TIME:{}'.format(time.asctime(time.localtime(time.time()))))
    logging.info('Training time: {:.2f} hours'.format(round((datetime.now().replace(microsecond=0)-start_time).seconds/3600, 2)))


def cal_loss(model, inputs, targets, criterions, epoch, with_acc=False):
    ce_criterion, rank_criterion, dynamic_criterion, mse_criterion = criterions
    acc = 0
    if args.nopfi:
        if model.has_cls_token:
            logits, global_feature, decision_mask_list, _part_masks, global_patch_features, decision_mask = model(inputs)
        else:
            logits, global_feature, decision_mask_list, _part_masks = model(inputs)
    else:
        if model.has_cls_token:
            logits, pfi_logits, pfi_targets, self_scores, other_scores, global_feature, decision_mask_list, _part_masks, global_patch_features, decision_mask = model(inputs, targets)
        else:
            logits, pfi_logits, pfi_targets, self_scores, other_scores, global_feature, decision_mask_list, _part_masks = model(inputs, targets)
    loss = ce_criterion(logits, targets)  # softmax_loss
    loss_str = f"ce_loss:{loss.item()}, "
    if not args.nopfi:
        pfi_loss = 2 * ce_criterion(pfi_logits, pfi_targets)  # softmax_loss
        loss += pfi_loss
        loss_str += f"pfi_loss:{pfi_loss.item()}, "
        flags = torch.ones([self_scores.size(0),]).to(device)
        rank_loss = rank_criterion(self_scores, other_scores, flags)  # rank_loss # deprecated
        loss += rank_loss
        loss_str += f"rank_loss:{rank_loss.item()}, "
    if dynamic_criterion:
        if model.has_cls_token:
            dynamic_loss = dynamic_criterion(inputs, [global_feature, global_patch_features, decision_mask, decision_mask_list])
        else:
            dynamic_loss = dynamic_criterion(inputs, [global_feature, decision_mask_list])
        loss += dynamic_loss
        loss_str += f"dynamic_loss:{dynamic_loss.item()}"

    if with_acc:
        acc = utils.cal_accuracy(logits, targets)
    loss.backward()

    if torch.isnan(loss):
        logging.error("Nan is detected in total loss!")
        exit(-1)

    if with_acc:
        return loss, loss_str, acc
    else:
        return loss, loss_str


def train(scaler, train_loader, model, criterions, optimizers, epoch):
    model.train()
    batch_loss_list = []
    batch_acc_list = []
    total = len(train_loader)
    for i, (inputs, targets, filenames) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        [optimizer.zero_grad() for optimizer in optimizers]
        loss, loss_str, acc = cal_loss(model, inputs, targets, criterions, epoch, with_acc=True)  # including loss.backward()
        [optimizer.step() for optimizer in optimizers]

        if i % cfg.train.log_step == 0:
            logging.info("Trainning epoch:{}/{} batch:{}/{} loss:{:.6f} acc:{:.6f} loss_detail: {}".format(epoch+1, args.epochs, i+1, total, loss.item(), acc, loss_str))
        batch_loss_list.append(loss.item())
        batch_acc_list.append(acc)

    return np.mean(batch_loss_list), np.mean(batch_acc_list)


def validate(val_loader, model, ce_criterion, epoch):
    model.eval()  # switch to evaluate mode
    batch_loss_list = []
    batch_acc_list = []
    with torch.no_grad():
        total = len(val_loader)
        for i, (inputs, targets, filenames) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)  # B num_classes, B L num_classes
            if len(outputs.size()) == 1:
                outputs = torch.unsqueeze(outputs, dim=0)

            loss = ce_criterion(outputs, targets)
            batch_loss_list.append(loss.item())

            acc = utils.cal_accuracy(outputs, targets)
            batch_acc_list.append(acc)

            if i % cfg.train.log_step == 0:
                logging.info("Validating epoch:{}/{} batch:{}/{} loss:{:.6f} acc:{:.6f}".format(epoch+1, args.epochs, i+1, total, loss.item(), acc))
    return np.mean(batch_loss_list), np.mean(batch_acc_list)


def visualize(val_loader, model, part_attn_save_dir):
    model.eval()  # switch to evaluate mode
    with torch.no_grad():
        total = len(val_loader)
        for i, (inputs, targets, filenames) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            patch_features, decision_mask, parts_masks = model(inputs, flag="visual")
            B, L, D = parts_masks.shape
            size = int(math.sqrt(L))
            parts_masks = parts_masks[0].reshape(size, size, D).detach().cpu().numpy()
            rows = 3
            columns = 4
            fig, axs = plt.subplots(rows,columns)
            for row in range(rows):
                for col in range(columns):
                    idx = row*columns+col
                    if idx < D:
                        axs[row,col].imshow(parts_masks[:,:,idx], cmap="jet", vmin=0, vmax=1)
            plt.savefig(os.path.join(part_attn_save_dir, "{}.png".format(i)))
            plt.close()
            if i % cfg.train.log_step == 0:
                logging.info("Visualize: batch:{}/{}".format(i+1, total))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="experiment name")
    parser.add_argument("--config", type=str, required=True, help="config file path")
    parser.add_argument("--arch", type=str, required=True, choices=supported_arch, help="model architecture")
    parser.add_argument("--gpus", type=str, default="0", help="gpu ids, example: 0,1")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--sample_classes", type=int, default=2, help="sample n classes from all classes each time")
    parser.add_argument("--sample_images", type=int, default=10, help="sample n images from each classes each time")
    parser.add_argument("--img_size", type=int, default=224, help="image size")
    parser.add_argument("--window_size", type=int, default=7, help="image_size:224,window_size:7, image_size:384,window_size:12;")
    parser.add_argument("--weights_dir", type=str, default=None, help=".pth weights directory")
    parser.add_argument("--lr", type=float, default=None, help="backbone learning rate")
    parser.add_argument("--distill", action="store_true", help="use teacher model")
    parser.add_argument("--from_scratch", action="store_true", help="without pretrain weights")
    parser.add_argument("--finetune", action="store_true", help="train in finetuning mode")
    parser.add_argument("--nopfi", action="store_true", help="without pfi module")
    parser.add_argument("--nolocal", action="store_true", help="without local branch")
    parser.add_argument("--nodynamic", action="store_true", help="without dynamic design in global branch")
    parser.add_argument("--vis_mode", action="store_true", help="only visualize")

    parser.add_argument("--ratio_weight", type=int, default=None, help="if None, set 2 for vit, set 10 for swin.")

    parser.add_argument("--pruning_loc", type=int, default=6, help="pruning_loc: 2, 4, 6, 8, 10, 12, 14, 16")
    parser.add_argument("--base_keep_rate", type=float, default=0.5, help="base keep rate in DynamicSwin backbone")

    parser.add_argument("--gaussian_ksize", type=int, default=15, help="gaussian_ksize: 3, 5, 7, 9, 11, 13, 15, 17, 19, 21")
    parser.add_argument("--num_parts", type=int, default=8, help="number of parts in LocalBranch")
    parser.add_argument("--part_channels", type=int, default=16, help="number of channels for each part in LocalBranch")
    parser.add_argument("--local_from_stage", type=int, default=-1, help="[0,1,2,3] for SwinT or -1 for all architecture")
    parser.add_argument("--local_depth", type=int, default=3, help="number of blocks in LocalBranch")

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    base_cfg = OmegaConf.load(cfg.parent)
    cfg = OmegaConf.merge(cfg, base_cfg)

    if args.gaussian_ksize == 0:
        args.gaussian_ksize = None
    if args.lr:
        cfg.train.backbone_lr = args.lr
        cfg.train.others_lr = args.lr * 5
    if args.vis_mode:
        print("[in VISUALIZE mode]")
        assert args.weights_dir is not None, "args.weights_dir shouldn't be None."

    timestamp = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    if not args.vis_mode:
        model_best_path  = "./middle/models/{}-{}/best.pth".format(args.name, timestamp)
        model_last_path  = "./middle/models/{}-{}/last.pth".format(args.name, timestamp)
        params_save_path = "./middle/models/{}-{}/params.pth".format(args.name, timestamp)
        log_path = "./middle/logs/{}-{}.log".format(args.name, timestamp)
        history_save_path = "./middle/history/{}-{}.png".format(args.name, timestamp)
        os.makedirs("./middle/logs/", exist_ok=True)
        os.makedirs("./middle/models/{}-{}/".format(args.name, timestamp), exist_ok=True)
        os.makedirs("./middle/history/", exist_ok=True)
        logging.basicConfig(
            level="INFO",
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.FileHandler(log_path, mode='a'), logging.StreamHandler()]
        )
    else:
        logging.basicConfig(
            level="INFO",
            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler()]
        )
        # cfg.val.num_workers = 0
    part_attn_save_dir = "./middle/parts-attn/"
    os.makedirs("./middle/parts-attn/", exist_ok=True)
    logging.info(f"timestamp:{timestamp}")

    if torch.cuda.is_available() and args.gpus != "cpu":
        device = torch.device(f'cuda:{args.gpus}')
    else:
        device = torch.device("cpu")

    try:
        main()
    except Exception as e:
        logging.error(e)
        logging.error(traceback.format_exc())
        exit(1)