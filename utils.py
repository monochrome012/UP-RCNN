import json
import logging
import os
import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops.boxes import box_iou


def setup_args():
    name = "UP-RCNN"
    parser = ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        default=name,
    )
    parser.add_argument(
        "--load_epoch",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--warm_up_epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--loss_weight",
        type=json.loads,
        default={
            "rpn_delta_loss": 0.1,
            "rpn_score_loss": 0.1,
            "rcnn_delta_loss": 0.1,
            "rcnn_score_loss": 0.1,
            "contrast_loss": 1,
        },
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default=f"{os.path.join('data','COCO2017','train2017')}",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default=f"{os.path.join('data','COCO2017','val2017')}",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=f"{os.path.join('data','COCO2017','test2017')}",
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    args.checkpoint_dir = f"{os.path.join('workplace',args.name,'checkpoint')}"
    args.plt_dir = f"{os.path.join('workplace',args.name,'plt')}"
    args.log_dir = f"{os.path.join('workplace',args.name,'log')}"

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.plt_dir):
        os.makedirs(args.plt_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.checkpoint_file is None and args.load_epoch > 0:
        args.checkpoint_file = os.path.join(args.checkpoint_dir, str(args.load_epoch) + ".pth.tar")

    return args


def get_logger(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(message)s')
    fh = logging.FileHandler(os.path.join(args.log_dir, args.name + ".txt"), "w")
    sh = logging.StreamHandler()
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def move_device(x, device):
    if isinstance(x, list):
        return [move_device(i, device) for i in x]
    if isinstance(x, dict):
        return {k: move_device(v, device) for k, v in x.items()}
    else:
        return x.to(device)


@torch.inference_mode()
def average_precision(pred_boxes, pred_scores, ground_truth, iou_threshold=0.5):

    average_precisions = []
    epsilon = 1e-6

    for bboxes, scores, gt in zip(pred_boxes, pred_scores, ground_truth):

        len_gt = len(gt)
        gt_mask = torch.zeros(len_gt)

        detections = zip(bboxes, scores)
        detections = sorted(detections, key=lambda x: x[1], reverse=True)
        if len(detections) <= 0:
            average_precisions.append(torch.zeros(1))
            continue

        bboxes = torch.stack([d[0] for d in detections])

        TP = torch.zeros(len(bboxes))
        FP = torch.zeros(len(bboxes))

        ious = box_iou(bboxes, gt)

        ious_max = torch.max(ious, dim=1)
        ious_max_val = ious_max.values
        ious_max_ind = ious_max.indices
        for d, (v, i) in enumerate(zip(ious_max_val, ious_max_ind)):
            if v.item() > iou_threshold and gt_mask[i] == 0:
                TP[d] = 1
                gt_mask[i] = 1
            else:
                FP[d] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / (len_gt + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def update(L, l):
    for key, value in l.items():
        if key in L.keys():
            L[key].append(np.mean(value))
        else:
            L[key] = [np.mean(value)]


def draw(l, dir, name):
    plt.title(name)
    for key in l:
        plt.semilogy(range(1, len(l[key]) + 1), l[key], label=key)
        plt.legend()
    plt.savefig(os.path.join(dir, name + ".png"), dpi=200)
    plt.close()
    for key in l:
        plt.title(name + "_" + key)
        plt.semilogy(range(1, len(l[key]) + 1), l[key], label=key)
        plt.legend()
        plt.savefig(os.path.join(dir, name + "_" + key + ".png"), dpi=200)
        plt.close()
