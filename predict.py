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

    model = UP_RCNN().cuda()

    map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}
    parameters = torch.load("./workplace/UP-RCNN/checkpoint/200.pth.tar", map_location=map_location)
    model.load_state_dict(parameters["model"])

    val_dataset = PatchDataset(dir=args.val_dir)
    # val_dataset = CopyPasteDataset(dir=args.val_dir)
    val_loader = DataLoader(dataset=val_dataset, batch_size=32, num_workers=1, collate_fn=val_dataset.collate_fn, shuffle=False, drop_last=True)

    model = model.eval()
    with torch.inference_mode():
        for i, batch_inputs in enumerate(val_loader):

            batch_inputs = move_device(batch_inputs, torch.device("cuda"))
            batch_outputs = model(batch_inputs)

            boxes_q = [input["boxes_q"].cuda() for input in batch_inputs]
            ap = average_precision(batch_outputs["pred_boxes"], batch_outputs["pred_scores"], boxes_q)
            print(f"Average Precision: {ap}")
            pred_boxes = batch_outputs["pred_boxes"]
            pred_scores = batch_outputs["pred_scores"]

            for i in range(len(batch_inputs)):
                image_q = postprocess(batch_inputs[i]["image_q"])
                images_s = [postprocess(p["image_s"]) for p in batch_inputs[i]["pool"]]

                # for j in range(len(images_s)):
                #     cv2.imwrite(f"./workplace/UP-RCNN/predict/{i}_{j}.jpg", images_s[j])

                for j in range(len(boxes_q[i])):
                    b = boxes_q[i][j]
                    x1 = int(b[0].item())
                    y1 = int(b[1].item())
                    x2 = int(b[2].item())
                    y2 = int(b[3].item())
                    cv2.rectangle(image_q, (x1, y1), (x2, y2), (255, 0, 0))

                for j in range(len(pred_boxes[i])):
                    b = pred_boxes[i][j]
                    s = pred_scores[i][j]
                    x1 = int(b[0].item())
                    y1 = int(b[1].item())
                    x2 = int(b[2].item())
                    y2 = int(b[3].item())
                    cv2.rectangle(image_q, (x1, y1), (x2, y2), (0, 0, 255))
                    text = "{:.3f}".format(s.item())
                    cv2.putText(image_q, text, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX, fontScale=0.4, color=(0, 0, 255))

                cv2.imwrite(f"./workplace/UP-RCNN/predict/{i}.jpg", image_q)
            break


if __name__ == "__main__":
    main()
