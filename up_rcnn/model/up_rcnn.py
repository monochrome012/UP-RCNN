import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

from .backbone import ResNet
from .contrast_head import ContrastHead
from .neck import FPN
from .rcnn_head import RCNNHead
from .rpn import RPN


class UP_RCNN(torch.nn.Module):

    def __init__(self, channel=256, K=65536, m=0.999, T=0.07):
        super(UP_RCNN, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        self.backbone_q = ResNet(50)
        self.backbone_k = ResNet(50)
        self.neck_q = FPN()
        self.neck_k = FPN()
        self.contrast_head_q = ContrastHead(channel)
        self.contrast_head_k = ContrastHead(channel)

        self.rpn = RPN(channel)
        self.rcnn_head = RCNNHead(channel)

        for param_q, param_k in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.neck_q.parameters(), self.neck_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.contrast_head_q.parameters(), self.contrast_head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(128, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, inputs):

        num_shots = len(inputs[0]["pool"])

        images_s = torch.cat([torch.stack([s["image_s"] for s in input["pool"]]) for input in inputs])  # tensor(B*S, C, H, W)
        images_q = torch.stack([input["image_q"] for input in inputs])  # tensor(B, C, H, W)
        images_k = torch.stack([input["image_k"] for input in inputs])  # tensor(B, C, H, W)

        boxes_s = [s["boxes_s"] for input in inputs for s in input["pool"]]  # list(B*S, tensor(1, 4))
        boxes_q = [input["boxes_q"] for input in inputs] if self.training else None  # list(B, tensor(1, 4))
        boxes_k = [input["boxes_k"] for input in inputs] if self.training else None  # list(B, tensor(1, 4))

        feats_s = self.neck_q(*self.backbone_q(images_s))  # list[F, tensor(B*S, C, H, W)]
        feats_q = self.neck_q(*self.backbone_q(images_q))  # list[F, tensor(B, C, H, W)]

        if self.training:
            box_feats_q = self.box_roi_pool(feats_q, boxes_q, images_q.shape[-2:])  # list[B, tensor(1, C, H, W)]
            logits_q = self.contrast_head_q(box_feats_q)  # tensor(B, 128)
            with torch.no_grad():
                self._momentum_update_key_encoder()
                images_k, boxes_k, idx_unshuffle = self._batch_shuffle_ddp(images_k, torch.stack(boxes_k))
                boxes_k = [b for b in boxes_k]
                feats_k = self.neck_k(*self.backbone_k(images_k))  # list[F, tensor(B, C, H, W)]
                box_feats_k = self.box_roi_pool(feats_k, boxes_k, images_k.shape[-2:])  # list[B, tensor(1, C, H, W)]
                logits_k = self.contrast_head_k(box_feats_k)  # tensor(B, 128)
                logits_k = self._batch_unshuffle_ddp(logits_k, idx_unshuffle)

        box_feats_s = self.box_roi_pool(feats_s, boxes_s, images_s.shape[-2:])  # list[B*S, tensor(1, C, H, W)]
        box_feats_pool_s = torch.stack(torch.split(torch.cat(box_feats_s).mean([2, 3], keepdim=True),
                                                   num_shots)).mean(1, keepdim=False)  # tensor(B, C, 1, 1)

        corattention = [
            torch.cat([F.conv2d(q.unsqueeze(0), s.unsqueeze(0).permute(1, 0, 2, 3), groups=q.shape[0]) for q, s in zip(f_q, box_feats_pool_s)])
            for f_q in feats_q
        ]  # list(F, tensor(B, C, H, W))

        proposals, rpn_loss = self.rpn(corattention, images_q.shape[-2:], boxes_q)  # list(B, tensor(N, 4))
        pred_box_feats_q = self.box_roi_pool(feats_q, proposals, images_q.shape[-2:])  # list(B, tensor(N, C, H, W))
        pred_boxes, pred_scores, rcnn_loss = self.rcnn_head(pred_box_feats_q, box_feats_s, proposals, images_q.shape[-2:],
                                                            boxes_q)  # list(B, tensor(M, 4)), list(B, tensor(M, 1))

        outputs = dict()
        if self.training:
            outputs.update(rpn_loss)
            outputs.update(rcnn_loss)
            outputs["contrast_loss"] = self._contrast(logits_q, logits_k)
        outputs["pred_boxes"] = pred_boxes
        outputs["pred_scores"] = pred_scores
        return outputs

    def box_roi_pool(self, feats, boxes, image_size, output_size=(7, 7)):
        num_boxes_per_image = [len(b) for b in boxes]
        num_levels = len(feats)

        concat_boxes = torch.cat(boxes, dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = torch.cat([torch.full_like(b[:, :1], i, dtype=dtype, layout=torch.strided, device=device) for i, b in enumerate(boxes)], dim=0)
        boxes = torch.cat([ids, concat_boxes], dim=1)

        levels = self._map_levels(boxes, num_levels, image_size)

        num_boxes = len(boxes)
        num_channels = feats[0].shape[1]

        dtype, device = feats[0].dtype, feats[0].device
        result = torch.zeros((num_boxes, num_channels) + output_size, dtype=dtype, device=device)

        for level in range(num_levels):
            idx_in_level = torch.where(levels == level)[0]
            boxes_per_level = boxes[idx_in_level]

            result_idx_in_level = roi_align(feats[level], boxes_per_level, output_size, feats[level].shape[-1] / image_size[-1])
            result[idx_in_level] = result_idx_in_level.to(result.dtype)

        result = result.split(num_boxes_per_image, 0)
        return result

    def _map_levels(self, boxes, num_levels, image_size):
        scale = torch.sqrt((boxes[:, 3] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 2]))
        levels = torch.floor(torch.log2(scale / (image_size[0] * image_size[1]) + 1e-6))
        levels = levels.clamp(min=0, max=num_levels - 1).long()
        return levels

    def _contrast(self, q, k):
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1)

        logits /= self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        self._dequeue_and_enqueue(k)

        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, y):
        batch_size_this = x.shape[0]

        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(y)
        assert (y_gather.shape[0] == x_gather.shape[0])

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        idx_shuffle = torch.randperm(batch_size_all).cuda()

        torch.distributed.broadcast(idx_shuffle, src=0)

        idx_unshuffle = torch.argsort(idx_shuffle)

        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], y_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)
        for param_q, param_k in zip(self.neck_q.parameters(), self.neck_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)
        for param_q, param_k in zip(self.contrast_head_q.parameters(), self.contrast_head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
