import torch


class BoxCoder:

    def __init__(self):
        super().__init__()

    def encode(self, orig_boxes, base_boxes):
        num_boxes_per_image = [len(b) for b in base_boxes]

        base_boxes = torch.cat(base_boxes, dim=0)
        orig_boxes = torch.cat(orig_boxes, dim=0)

        base_w = base_boxes[:, 2] - base_boxes[:, 0]
        base_h = base_boxes[:, 3] - base_boxes[:, 1]
        base_ctr_x = base_boxes[:, 0] + 0.5 * base_w
        base_ctr_y = base_boxes[:, 1] + 0.5 * base_h

        src_w = orig_boxes[:, 2] - orig_boxes[:, 0]
        src_h = orig_boxes[:, 3] - orig_boxes[:, 1]
        src_ctr_x = orig_boxes[:, 0] + 0.5 * src_w
        src_ctr_y = orig_boxes[:, 1] + 0.5 * src_h

        dx = ((src_ctr_x - base_ctr_x) / base_w).unsqueeze(1)
        dy = ((src_ctr_y - base_ctr_y) / base_h).unsqueeze(1)
        dw = (torch.log(src_w / base_w)).unsqueeze(1)
        dh = (torch.log(src_h / base_h)).unsqueeze(1)

        deltas = torch.cat((dx, dy, dw, dh), dim=1)
        deltas = deltas.split(num_boxes_per_image, 0)

        return deltas

    def decode(self, deltas, base_boxes):
        num_boxes_per_image = [len(b) for b in base_boxes]

        base_boxes = torch.cat(base_boxes, dim=0)
        deltas = deltas.reshape(base_boxes.shape[0], -1)

        base_w = base_boxes[:, 2] - base_boxes[:, 0]
        base_h = base_boxes[:, 3] - base_boxes[:, 1]
        base_ctr_x = base_boxes[:, 0] + 0.5 * base_w
        stc_ctr_y = base_boxes[:, 1] + 0.5 * base_h

        dx = deltas[:, 0::4]
        dy = deltas[:, 1::4]
        dw = deltas[:, 2::4]
        dh = deltas[:, 3::4]

        pred_ctr_x = dx * base_w[:, None] + base_ctr_x[:, None]
        pred_ctr_y = dy * base_h[:, None] + stc_ctr_y[:, None]
        pred_w = torch.exp(dw) * base_w[:, None]
        pred_h = torch.exp(dh) * base_h[:, None]

        boxes = torch.stack((pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h, pred_ctr_x + 0.5 * pred_w, pred_ctr_y + 0.5 * pred_h),
                            dim=2).flatten(1)
        boxes = boxes.split(num_boxes_per_image, 0)

        return boxes
