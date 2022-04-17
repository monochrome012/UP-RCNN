import random

import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import transforms


class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class BoxTrans(object):

    def __init__(self, image_size=(224, 224), val=False):
        self.image_size = image_size
        self.val = val

    def __call__(self, image, boxes=None):

        boxes = resize_bbox(boxes, image.shape[:2], self.image_size)
        image = np.array(transforms.Resize(self.image_size)(Image.fromarray(image)))

        if self.val is False:
            image, boxes = random_flip(image, self.image_size, boxes)

        return image, boxes


class ImageTrans(object):

    def __init__(self, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], val=False):
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)

        if val is True:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.trans = transforms.Compose([
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

    def __call__(self, image):
        image = self.trans(Image.fromarray(image))
        return image


def resize_bbox(boxes, in_size, out_size):
    x_scale = float(out_size[1]) / in_size[1]
    y_scale = float(out_size[0]) / in_size[0]
    boxes[:, 0] = x_scale * boxes[:, 0]
    boxes[:, 2] = x_scale * boxes[:, 2]
    boxes[:, 1] = y_scale * boxes[:, 1]
    boxes[:, 3] = y_scale * boxes[:, 3]
    return boxes


def random_flip(image, image_size, boxes):
    w_flip = random.choice([True, False])
    h_flip = random.choice([True, False])

    if w_flip:
        image = image[:, ::-1, :]
        w_max = image_size[1] - 1 - boxes[:, 0]
        w_min = image_size[1] - 1 - boxes[:, 2]
        boxes[:, 0] = w_min
        boxes[:, 2] = w_max
    if h_flip:
        image = image[::-1, :, :]
        h_max = image_size[0] - 1 - boxes[:, 1]
        h_min = image_size[0] - 1 - boxes[:, 3]
        boxes[:, 1] = h_min
        boxes[:, 3] = h_max
    return image, boxes
