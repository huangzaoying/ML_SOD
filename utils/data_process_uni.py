import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os


def preprocess_img(img_dir, channels=3):

    if channels == 1:
        img = cv2.imread(img_dir, 0)
    elif channels == 3:
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image_org = img
    shape_r = 256
    shape_c = 256
    img_padded = np.ones((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)
    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c
    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[
            :,
            ((img_padded.shape[1] - new_cols) // 2) : (
                (img_padded.shape[1] - new_cols) // 2 + new_cols
            ),
        ] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))

        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[
            ((img_padded.shape[0] - new_rows) // 2) : (
                (img_padded.shape[0] - new_rows) // 2 + new_rows
            ),
            :,
        ] = img

    return img_padded, image_org


def postprocess_img(pred, org_dir):
    pred = np.array(pred)
    org = cv2.imread(org_dir, 0)
    shape_r = org.shape[0]
    shape_c = org.shape[1]
    predictions_shape = pred.shape

    rows_rate = shape_r / predictions_shape[0]
    cols_rate = shape_c / predictions_shape[1]

    if rows_rate > cols_rate:
        new_cols = (predictions_shape[1] * shape_r) // predictions_shape[0]
        pred = cv2.resize(pred, (new_cols, shape_r))
        img = pred[
            :,
            ((pred.shape[1] - shape_c) // 2) : (
                (pred.shape[1] - shape_c) // 2 + shape_c
            ),
        ]
    else:
        new_rows = (predictions_shape[0] * shape_c) // predictions_shape[1]
        pred = cv2.resize(pred, (shape_c, new_rows))
        img = pred[
            ((pred.shape[0] - shape_r) // 2) : (
                (pred.shape[0] - shape_r) // 2 + shape_r
            ),
            :,
        ]
    return img


class SODDataset(Dataset):
    def __init__(self, base_dir, transform=None, num_classes=4):
        self.base_dir = base_dir
        self.transform = transform
        self.num_classes = num_classes

        # Collect data
        self.data = []
        self.categories = sorted(
            os.listdir(os.path.join(base_dir, "Stimuli"))
        )  # 类别列表
        for _, category in enumerate(self.categories):
            stim_path = os.path.join(base_dir, "Stimuli", category)
            fix_path = os.path.join(base_dir, "FIXATIONMAPS", category)

            if os.path.isdir(stim_path) and os.path.isdir(fix_path):
                for img_name in os.listdir(stim_path):
                    stim_img_path = os.path.join(stim_path, img_name)
                    fix_img_path = os.path.join(fix_path, img_name)
                    if os.path.isfile(stim_img_path) and os.path.isfile(fix_img_path):
                        self.data.append((stim_img_path, fix_img_path, 1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        stim_img_path, fix_img_path, label_idx = self.data[idx]

        # Load image
        image, _ = preprocess_img(stim_img_path, channels=3)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        saliency, _ = preprocess_img(fix_img_path, channels=1)
        saliency = np.array(saliency, dtype=np.float32) / 255.0
        saliency = torch.from_numpy(saliency).unsqueeze(0)
        # Convert label to one-hot vector
        label = torch.zeros(self.num_classes)
        label[label_idx] = 1
        sample = {
            "image": image,
            "saliency": saliency,
            "label": label,
        }
        return sample
