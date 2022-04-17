import torch

from .box_coder import BoxCoder


class BoxMatcher:

    def __init__(self, pos_iou_thresh=0.7, neg_iou_thresh=0.3):
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.box_coder = BoxCoder()

    def __call__(self, orig_boxes, base_boxes):
        labels = []
        matched_boxes = []
        for orig_boxes_per_image, base_boxes_per_image in zip(orig_boxes, base_boxes):
            labels_per_image = torch.empty((len(orig_boxes_per_image), ), dtype=torch.float32, device=orig_boxes_per_image.device)
            labels_per_image.fill_(-1)

            if orig_boxes_per_image.shape[0] <= 0:
                return labels_per_image, orig_boxes_per_image

            ious = self._box_iou(orig_boxes_per_image, base_boxes_per_image)
            argmax_ious = ious.argmax(axis=1)
            max_ious = ious[torch.arange(len(orig_boxes_per_image), device=argmax_ious.device), argmax_ious]
            gt_argmax_ious = ious.argmax(axis=0)

            labels_per_image[max_ious < self.neg_iou_thresh] = 0
            labels_per_image[gt_argmax_ious] = 1
            labels_per_image[max_ious >= self.pos_iou_thresh] = 1

            labels.append(labels_per_image)
            matched_boxes.append(base_boxes_per_image[argmax_ious])

        return labels, matched_boxes

    def _box_iou(self, boxes_1, boxes_2):
        tl = torch.maximum(boxes_1[:, None, :2], boxes_2[None, :, :2])
        br = torch.minimum(boxes_1[:, None, 2:], boxes_2[None, :, 2:])

        area_i = torch.prod(br - tl, axis=2) * (tl < br).all(axis=2)
        area_a = torch.prod(boxes_1[:, 2:] - boxes_1[:, :2], axis=1)
        area_b = torch.prod(boxes_2[:, 2:] - boxes_2[:, :2], axis=1)
        return area_i / (area_a[:, None] + area_b[None, :] - area_i)
