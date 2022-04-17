import torch
from torchvision.ops import nms

from .box_coder import BoxCoder


class BoxFilter:

    def __init__(self, pre_nms=1000, pos_nms=1000, nms_thresh=0.7, scores_thresh=0.5, min_size=16):
        self.pre_nms = pre_nms
        self.pos_nms = pos_nms
        self.nms_thresh = nms_thresh
        self.score_thresh = scores_thresh
        self.min_size = min_size
        self.box_coder = BoxCoder()

    def __call__(self, boxes, scores, image_size):
        boxes_filtered, scores_filtered = [], []
        for boxes_per_image, scores_per_image in zip(boxes, scores):
            dim = boxes_per_image.dim()
            boxes_x = boxes_per_image[:, 0::2]
            boxes_y = boxes_per_image[:, 1::2]
            boxes_x = boxes_x.clamp(min=0, max=image_size[1] - 1)
            boxes_y = boxes_y.clamp(min=0, max=image_size[0] - 1)
            boxes_per_image = torch.stack((boxes_x, boxes_y), dim=dim).reshape(boxes_per_image.shape)

            ws = boxes_per_image[:, 2] - boxes_per_image[:, 0]
            hs = boxes_per_image[:, 3] - boxes_per_image[:, 1]
            keep = (ws >= self.min_size) & (hs >= self.min_size)
            keep = torch.where(keep)[0]
            boxes_per_image = boxes_per_image[keep]
            scores_per_image = scores_per_image[keep]

            keep = torch.where(scores_per_image > self.score_thresh)[0]
            boxes_per_image = boxes_per_image[keep]
            scores_per_image = scores_per_image[keep]

            order = scores_per_image.view(-1).argsort()
            if self.pre_nms > 0:
                order = order[:self.pre_nms]
            boxes_per_image = boxes_per_image[order]
            scores_per_image = scores_per_image[order]

            keep = nms(boxes_per_image, scores_per_image.view(-1), self.nms_thresh)

            if self.pos_nms > 0:
                keep = keep[:self.pos_nms]
            boxes_per_image = boxes_per_image[keep]
            scores_per_image = scores_per_image[keep]

            boxes_filtered.append(boxes_per_image)
            scores_filtered.append(scores_per_image)
        return boxes_filtered, scores_filtered
