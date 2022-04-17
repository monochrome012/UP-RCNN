import math
import os
import random

import cv2
import selectivesearch
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from .utils import BoxTrans, ImageTrans


class PatchDataset(Dataset):

    def __init__(self, dir, cut_scale=(0.2, 1), paste_scale=(0.2, 1), ratio=(3. / 4., 4. / 3.), image_size=(224, 224), len_dataset=None):
        self.paths = [os.path.join(dir, basename) for basename in sorted(os.listdir(dir))]
        self.len = len(self.paths) if len_dataset is None else len_dataset
        self.cut_scale = cut_scale
        self.paste_scale = paste_scale
        self.ratio = ratio
        self.box_trans = BoxTrans(image_size)
        self.image_trans = ImageTrans()
        self.image_size = image_size

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        image_q = cv2.imread(random.choice(self.paths))

        x, y, w, h = self.get_position(image_q, self.cut_scale)
        boxes_q = [[x, y, x + w - 1, y + h - 1]]

        image_s = image_q[y:y + h, x:x + w, :]
        boxes_s = [[0, 0, w - 1, h - 1]]

        image_s, boxes_s = self.box_trans(image_s, torch.tensor(boxes_s, dtype=torch.float32))
        image_q, boxes_q = self.box_trans(image_q, torch.tensor(boxes_q, dtype=torch.float32))

        image_s = self.image_trans(image_s)
        image_q = self.image_trans(image_q)

        datas = {
            "pool": [{
                "image_s": image_s,
                "boxes_s": boxes_s,
            }],
            "image_q": image_q,
            "boxes_q": boxes_q,
            "image_k": image_s,
            "boxes_k": boxes_s,
        }

        return datas

    def _get_position(self, image, scale):
        img_lbl, regions = selectivesearch.selective_search(image, scale=200, min_size=32)
        height = image.shape[0]
        width = image.shape[1]
        area = height * width

        for r in regions:
            x, y, w, h = r['rect']
            if r['size'] < min(scale) * area or r['size'] > max(scale) * area:
                continue
            if w / h < min(self.ratio) or w / h > max(self.ratio):
                continue
            return x, y, w, h
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:
            w = width
            h = height
        x = (width - w) // 2
        y = (height - h) // 2
        return x, y, w, h

    def get_position(self, image, scale):
        width = image.shape[1]
        height = image.shape[0]
        area = height * width
        log_ratio = torch.log(torch.tensor(self.ratio))
        log_scale = torch.log(torch.tensor(scale))
        for _ in range(100):
            # target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            target_area = area * torch.exp(torch.empty(1).uniform_(log_scale[0], log_scale[1])).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                x = random.randint(0, width - w)
                y = random.randint(0, height - h)
                return x, y, w, h

        in_ratio = float(width) / float(height)
        if in_ratio < min(self.ratio):
            w = width
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = height
            w = int(round(h * max(self.ratio)))
        else:
            w = width
            h = height
        x = (width - w) // 2
        y = (height - h) // 2
        return x, y, w, h

    def get_aug(self, image_shape):
        aug = transforms.Compose([
            transforms.RandomResizedCrop(image_shape, scale=(0.8, 1.)),
            # transforms.RandomRotation(360),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
        ])
        return aug

    def collate_fn(self, batch):
        return batch
