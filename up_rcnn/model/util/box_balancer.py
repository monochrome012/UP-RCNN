import torch


class BoxBalancer:

    def __init__(self, num_sample=256, pos_ratio=0.5):
        self.num_sample = num_sample
        self.pos_ratio = pos_ratio

    def __call__(self, labels):
        pos_mask, neg_mask = [], []

        for labels_per_image in labels:
            positive = torch.where(labels_per_image >= 1)[0]
            negative = torch.where(labels_per_image == 0)[0]

            num_pos = int(self.pos_ratio * self.num_sample)
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.num_sample - num_pos
            num_neg = min(negative.numel(), num_neg)

            perm_pos = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm_neg = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm_pos]
            neg_idx_per_image = negative[perm_neg]

            pos_mask_per_image = torch.zeros_like(labels_per_image, dtype=torch.uint8)
            neg_mask_per_imagg = torch.zeros_like(labels_per_image, dtype=torch.uint8)

            pos_mask_per_image[pos_idx_per_image] = 1
            neg_mask_per_imagg[neg_idx_per_image] = 1

            pos_mask.append(pos_mask_per_image)
            neg_mask.append(neg_mask_per_imagg)
            # print(self.pos_ratio, len(pos_idx_per_image), len(neg_idx_per_image))

        pos_inds = torch.where(torch.cat(pos_mask, dim=0))[0]
        neg_inds = torch.where(torch.cat(neg_mask, dim=0))[0]
        sampled_inds = torch.cat([pos_inds, neg_inds], dim=0)

        return pos_inds, neg_inds, sampled_inds
