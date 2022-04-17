import torch
import torch.nn.functional as F
from torch import nn

from .util import AnchorGenerator, BoxBalancer, BoxCoder, BoxFilter, BoxMatcher


class RPNHead(nn.Module):

    def __init__(self, in_channels, num_anchors):
        super(RPNHead, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.score_predictor = nn.Conv2d(in_channels, num_anchors, 1, 1, 0)
        self.delta_predictor = nn.Conv2d(in_channels, num_anchors * 4, 1, 1, 0)

    def forward(self, feats):
        pred_deltas = []
        pred_scores = []
        for feats_per_level in feats:
            t = F.relu(self.conv(feats_per_level))
            pred_scores.append(self.score_predictor(t))
            pred_deltas.append(self.delta_predictor(t))
        return pred_scores, pred_deltas


class RPN(torch.nn.Module):

    def __init__(self, in_channels):
        super(RPN, self).__init__()

        self.anchor_generator = AnchorGenerator()
        self.box_balancer = BoxBalancer(num_sample=8, pos_ratio=0.5)
        self.box_coder = BoxCoder()
        self.box_filter = BoxFilter(scores_thresh=0)
        self.box_matcher = BoxMatcher()
        self.rpn_head = RPNHead(in_channels, len(self.anchor_generator.cell_anchors))

    def forward(self, feats, image_size, target_boxes=None):

        num_images = feats[0].shape[0]
        anchors = self.anchor_generator(feats, image_size)  # list(B, tensor(F*H*W*A, 4))

        pred_scores, pred_deltas = self.rpn_head(feats)  # list(F, tensor(B, A, H, W))  list(F, tensor(B, A*4, H, W))
        pred_scores, pred_deltas = self._concat_layers(pred_scores, pred_deltas, num_images)  # tensor(B, F*H*W*A, 1)  tensor(B, F*H*W*A, 4)

        pred_boxes = self.box_coder.decode(pred_deltas.detach(), anchors)  # list(B, tensor(F*H*W*A, 4))
        pred_scores_sigmoid = [torch.sigmoid(s.detach()) for s in pred_scores]  # list(B, tensor(F*H*W*A))
        proposals, _ = self.box_filter(pred_boxes, pred_scores_sigmoid, image_size)  # list(B, tensor(N, 4))

        loss = dict()
        if target_boxes is not None:
            labels, matched_boxes = self.box_matcher(anchors, target_boxes)  # list(B, tensor(F*H*W*A))  list(B, tensor(F*H*W*A, 4))
            matched_deltas = self.box_coder.encode(matched_boxes, anchors)  # list(B, tensor(F*H*W*A, 4))

            pos_idx, neg_idx, sample_idx = self.box_balancer(labels)  # tensor(P,)  tensor(N,)  tensor(P+N,)

            labels = torch.cat(labels, dim=0)  # tensor(B*F*H*W*A)
            matched_deltas = torch.cat(matched_deltas, dim=0)  # tensor(B*F*H*W*A, 4)
            pred_scores = pred_scores.reshape(-1)  # tensor(B*F*H*W*A)
            pred_deltas = pred_deltas.reshape(-1, 4)  # tensor(B*F*H*W*A, 4)

            delta_loss = F.smooth_l1_loss(pred_deltas[pos_idx], matched_deltas[pos_idx], beta=1 / 9, reduction="sum") / sample_idx.shape[0]
            score_loss = F.binary_cross_entropy_with_logits(pred_scores[sample_idx], labels[sample_idx])

            loss["rpn_delta_loss"] = delta_loss
            loss["rpn_score_loss"] = score_loss

        return proposals, loss

    def _concat_layers(self, pred_scores, pred_deltas, num_images):
        pred_scores_flattened = []
        pred_deltas_flattened = []

        for pred_scores_per_level, pred_deltas_per_level in zip(pred_scores, pred_deltas):
            B, A, H, W = pred_scores_per_level.shape

            pred_scores_per_level = self._permute_and_flatten(pred_scores_per_level, B, A, 1, H, W)
            pred_deltas_per_level = self._permute_and_flatten(pred_deltas_per_level, B, A, 4, H, W)

            pred_scores_flattened.append(pred_scores_per_level)
            pred_deltas_flattened.append(pred_deltas_per_level)

        pred_scores_concated = torch.cat(pred_scores_flattened, dim=1)
        pred_deltas_concated = torch.cat(pred_deltas_flattened, dim=1)

        return pred_scores_concated, pred_deltas_concated

    def _permute_and_flatten(self, layer, B, A, C, H, W):
        layer = layer.view(B, A, C, H, W)
        layer = layer.permute(0, 3, 4, 1, 2)
        layer = layer.reshape(B, -1, C)
        return layer
