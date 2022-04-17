import torch
import torch.nn.functional as F
from torch import nn

from .util import BoxBalancer, BoxCoder, BoxFilter, BoxMatcher


class RCNNHead(nn.Module):

    def __init__(self, in_channel, box_roi_pool=None):
        super(RCNNHead, self).__init__()

        self.box_balancer = BoxBalancer(num_sample=16, pos_ratio=0.25)
        self.box_coder = BoxCoder()
        self.box_filter = BoxFilter(nms_thresh=0.5)
        self.box_matcher = BoxMatcher(pos_iou_thresh=0.5, neg_iou_thresh=0.5)

        self.conv1 = nn.Conv2d(in_channel * 2, int(in_channel / 4), 1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(int(in_channel / 4), int(in_channel / 4), 3, padding=0, bias=False)
        self.conv3 = nn.Conv2d(int(in_channel / 4), in_channel, 1, padding=0, bias=False)
        self.delta_predictor = nn.Linear(in_channel, 4)
        self.score_predictor = nn.Linear(in_channel, 1)

        self.conv_cor = nn.Conv2d(in_channel, in_channel, 1, padding=0, bias=False)
        self.score_cor = nn.Linear(in_channel, 1)

        self.fc1 = nn.Linear(in_channel * 2, in_channel)
        self.fc2 = nn.Linear(in_channel, in_channel)
        self.score_fc = nn.Linear(in_channel, 1)

        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1)
        self.avgpool_fc = nn.AvgPool2d(7)

    def forward(self, box_feats_q, box_feats_s, proposals, image_size, target_boxes=None):

        x_q = torch.cat(box_feats_q)  # tensor(B*N, C, H, W)
        x_s = torch.cat([s.expand_as(q) for q, s in zip(box_feats_q, box_feats_s)])  # tensor(B*N, C, H, W)

        # global_relation
        x_fc_q = self.avgpool_fc(x_q).squeeze(3).squeeze(2)  # tensor(B*N, C)
        x_fc_s = self.avgpool_fc(x_s).squeeze(3).squeeze(2)  # tensor(B*N, C)
        cat_fc = torch.cat((x_fc_q, x_fc_s), 1)  # tensor(B*N, 2C)
        out_fc = F.relu(self.fc1(cat_fc), inplace=True)  # tensor(B*N, C)
        out_fc = F.relu(self.fc2(out_fc), inplace=True)  # tensor(B*N, C)
        score_fc = self.score_fc(out_fc)  # tensor(B*N, 1)

        # correlation
        x_cor_q = self.conv_cor(x_q)  # tensor(B*N, C, H, W)
        x_cor_s = self.conv_cor(x_s)  # tensor(B*N, C, H, W)
        x_cor = torch.cat([F.conv2d(q.unsqueeze(0), s.unsqueeze(0).permute(1, 0, 2, 3), groups=q.shape[0])
                           for q, s in zip(x_cor_q, x_cor_s)])  # tensor(B*N, C, H, W)
        x_relu = F.relu(x_cor, inplace=True).squeeze(3).squeeze(2)  # tensor(B*N, C)
        score_cor = self.score_cor(x_relu)  # tensor(B*N, 1)

        # relation
        x = torch.cat((x_q, x_s), 1)  # tensor(B*N, C*2, H, W)
        x = F.relu(self.conv1(x), inplace=True)  # tensor(B*N, C/2, H, W)
        x = self.avgpool(x)  # tensor(B*N, C/2, H-2, W-2)
        x = F.relu(self.conv2(x), inplace=True)  # tensor(B*N, C/2, H-4, W-4)
        x = F.relu(self.conv3(x), inplace=True)  # tensor(B*N, C/2, H-4, W-4)
        x = self.avgpool(x)  # tensor(B*N, C/2, H-6, W-6)
        x = x.squeeze(3).squeeze(2)  # tensor(B*N, C/2)
        score_pr = self.score_predictor(x)  # tensor(B*N, 1)

        pred_scores = score_pr + score_cor + score_fc  # tensor(B*N, 1)
        pred_deltas = self.delta_predictor(x)  # tensor(B*N, 4)

        num_boxes_per_image = [len(b) for b in proposals]
        pred_scores = pred_scores.split(num_boxes_per_image, 0)  # list(B, tensor(N, 1))

        pred_boxes = self.box_coder.decode(pred_deltas.detach(), proposals)  #  list(B, tensor(N, 4))

        pred_scores_sigmoid = [torch.sigmoid(s.detach()) for s in pred_scores]  # list(B, tensor(N, 1))
        boxes, scores = self.box_filter(pred_boxes, pred_scores_sigmoid, image_size)  # list(B, tensor(N, 4))  list(B, tensor(N, 1))

        loss = dict()
        if target_boxes is not None:
            labels, matched_boxes = self.box_matcher(proposals, target_boxes)  # list(B, tensor(N))  list(B, tensor(N, 4))
            matched_deltas = self.box_coder.encode(matched_boxes, proposals)  # list(B, tensor(N, 4))

            pos_idx, neg_idx, sample_idx = self.box_balancer(labels)  # tensor(P,)  tensor(N,)  tensor(P+N,)

            labels = torch.cat(labels, dim=0)  # tensor(B*N)
            matched_deltas = torch.cat(matched_deltas, dim=0)  # tensor(B*N, 4)
            pred_scores = torch.cat(pred_scores, dim=0).squeeze(-1)  # tensor(B*N)

            delta_loss = F.smooth_l1_loss(pred_deltas[pos_idx], matched_deltas[pos_idx], beta=1 / 9, reduction="sum") / sample_idx.shape[0]
            score_loss = F.binary_cross_entropy_with_logits(pred_scores[sample_idx], labels[sample_idx])
            loss["rcnn_delta_loss"] = delta_loss
            loss["rcnn_score_loss"] = score_loss

            return boxes, scores, loss

        return boxes, scores, loss
