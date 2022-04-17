import torch


class AnchorGenerator:

    def __init__(self, scales=[48, 96, 192], ratios=[0.5, 1, 2], device=torch.device("cuda")):
        super().__init__()
        self.scales = torch.tensor(scales, device=device)
        self.ratios = torch.tensor(ratios, device=device)
        self.cell_anchors = self.generate_anchors()

    def generate_anchors(self):
        w_ratios = torch.sqrt(self.ratios)
        h_ratios = 1 / w_ratios
        ws = (w_ratios[:, None] * self.scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * self.scales[None, :]).view(-1)
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        return base_anchors.round()

    def grid_anchors(self, grid_size, image_size, device):
        grid_height, grid_width = grid_size
        self.cell_anchors.to(device)

        shifts_x = torch.arange(0, grid_width, dtype=torch.int32, device=device) * image_size[1] / grid_width
        shifts_y = torch.arange(0, grid_height, dtype=torch.int32, device=device) * image_size[0] / grid_height

        shift_x, shift_y = torch.meshgrid(shifts_x, shifts_y, indexing="ij")
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = torch.stack((shift_x, shift_x, shift_y, shift_y), dim=1)

        anchors = (shifts.view(-1, 1, 4) + self.cell_anchors.view(1, -1, 4)).reshape(-1, 4)

        return anchors

    def __call__(self, feats, image_size):
        anchors_per_image = [self.grid_anchors(f.shape[-2:], image_size, feats[0].device) for f in feats]
        anchors = []
        for _ in range(feats[0].shape[0]):
            anchors.append(torch.cat(anchors_per_image))
        return anchors
