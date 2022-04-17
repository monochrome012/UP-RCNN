import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from up_rcnn.data import CopyPasteDataset, PatchDataset
from up_rcnn.model import UP_RCNN
from utils import average_precision, move_device, setup_args, setup_seed


def postprocess(image):
    mean = [123.675, 116.28, 103.53],
    std = [58.395, 57.12, 57.375],
    image = torch.clamp(image.permute(1, 2, 0) * torch.tensor(std, device=image.device) + torch.tensor(mean, device=image.device), 0, 1)
    return np.array(transforms.ToPILImage()(image.permute(2, 0, 1)))


def main():

    setup_seed(0)
    args = setup_args()

    val_dataset = CopyPasteDataset(dir=args.val_dir)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=32,
                            num_workers=args.num_workers,
                            collate_fn=val_dataset.collate_fn,
                            shuffle=False,
                            drop_last=True)

    with torch.inference_mode():
        for i, batch_inputs in enumerate(val_loader):

            for i in range(len(batch_inputs)):
                images_s = [postprocess(p["image_s"]) for p in batch_inputs[i]["pool"]]
                image_q = postprocess(batch_inputs[i]["image_q"])

                for j in range(len(images_s)):
                    cv2.imwrite(f"./workplace/UP-RCNN/test/{i}_{j}.jpg", images_s[j])

                for j in range(len(batch_inputs[i]["boxes_q"])):
                    b = batch_inputs[i]["boxes_q"][j]
                    x1 = int(b[0].item())
                    y1 = int(b[1].item())
                    x2 = int(b[2].item())
                    y2 = int(b[3].item())
                    cv2.rectangle(image_q, (x1, y1), (x2, y2), (0, 0, 255))
                cv2.imwrite(f"./workplace/UP-RCNN/test/{i}_q.jpg", image_q)

                image_k = postprocess(batch_inputs[i]["image_k"])
                for j in range(len(batch_inputs[i]["boxes_k"])):
                    b = batch_inputs[i]["boxes_k"][j]
                    x1 = int(b[0].item())
                    y1 = int(b[1].item())
                    x2 = int(b[2].item())
                    y2 = int(b[3].item())
                    cv2.rectangle(image_k, (x1, y1), (x2, y2), (0, 0, 255))
                cv2.imwrite(f"./workplace/UP-RCNN/test/{i}_k.jpg", image_k)
            break


if __name__ == "__main__":
    main()
