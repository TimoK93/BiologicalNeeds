"""
Author: Katharina Löffler (2022), Karlsruhe Institute of Technology
Licensed under MIT License
Modifications: removed unused imports, added type hints, removed unused code
    added multiscale, shifts and multisegmentation as arguments
"""
import os
import re
import cv2
import pickle

import shutil
from itertools import product
from embedtrack.utils.utils import get_img_files
import embedtrack.infer.utils as infer_utils
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from datetime import datetime
from embedtrack.models.net import TrackERFNet
import embedtrack.utils.transforms as my_transforms
from embedtrack.utils.utils import get_indices_pandas
from pathlib import Path
import io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InferenceDataSet(Dataset):
    """
    TwoDimensionalDataset class
    """

    def __init__(
        self,
        img_crops_curr_frame,
        img_crops_prev_frame,
        bg_id=0,
        transform=None,
    ):
        """
        Initialize dataset
        Args:
            img_crops_curr_frame: list
                list of np.arrays containing image crops from time step t
            img_crops_prev_frame: list
                list of np.arrays containing image crops from time step t-1
            bg_id:
            transform: Callable
             transformations to apply to each sample
        """

        # get image and instance list

        self.image_list = list(zip(img_crops_curr_frame, img_crops_prev_frame))
        # todo: remove??
        self.bg_id = bg_id
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def convert_yx_to_cyx(self, im, key):
        if im.ndim == 2 and key == "image":  # gray-scale image
            im = im[np.newaxis, ...]  # CYX
        elif im.ndim == 3 and key == "image":  # multi-channel image image
            pass
        else:
            im = im[np.newaxis, ...]
        return im

    def __getitem__(self, index):
        image_curr, image_prev = self.image_list[index]
        sample = dict()
        sample["image_curr"] = self.convert_yx_to_cyx(image_curr, key="image")
        sample["image_prev"] = self.convert_yx_to_cyx(image_prev, key="image")
        sample["index"] = index  # CYX
        # transform
        if self.transform is not None:
            return self.transform(sample)
        else:
            return sample

# use smooth patching idea https://github.com/Vooban/Smoothly-Blend-Image-Patches
def generate_crops(image_file, crop_size=256, overlap=0.25, scale=1.0):
    """
    Generate overlapping crops from an image-
    Args:
        image_file: string
            Path to the image to generate crops from
        crop_size: int
            Size of the squared shaped image crops
        overlap: float
            overlap between neighboring crops

    Returns:

    """
    image = tifffile.imread(image_file)
    if scale != 1.0:
        image = cv2.resize(
            image, (int(image.shape[1] * scale), int(image.shape[0] * scale))
        )
    _, (pad_h, pad_w) = calc_padded_img_size(image.shape, crop_size, overlap)
    image = np.pad(image, (pad_h, pad_w), mode="reflect")
    size_y, size_x = image.shape
    # 1-overlap so the overlap refers correctly to the overlapping area
    y_start = (np.arange(0, size_y - crop_size + 1, crop_size * (1 - overlap))).astype(
        int
    )
    x_start = (np.arange(0, size_x - crop_size + 1, crop_size * (1 - overlap))).astype(
        int
    )
    upper_left = list(product(y_start, x_start))
    crops = []
    for i, (y, x) in enumerate(upper_left):
        im_crop = image[y : y + crop_size, x : x + crop_size]
        if np.any(np.array(im_crop.shape) < crop_size):
            continue
        crops.append(im_crop)
    return crops


def patch_crops_gpu(
    image_crops,
    position_index,
    padded_img_size,
    img_size,
    window,
    crop_size=256,
    overlap=0.25,
):
    """
    Stitch predicted crops to a full image
    Args:
        image_crops: list
            list of predicted crops
        position_index: list
            list of crop positions
        padded_img_size: tuple
            size of the padded image
        img_size: tuple
            original image size
        window: Callable
            Smoothing function to apply to the crops
        crop_size: int
            size of the squared shaped crops
        overlap: float
            overlap between neighboring crops

    Returns: np.array (the stiched prediction)

    """
    if len(image_crops.shape) == 5:
        naugs = image_crops.shape[-1]
    else:
        naugs = 1
    dev = image_crops.device
    _image_size = padded_img_size if naugs == 1 else padded_img_size + (naugs,)
    weights = torch.zeros(_image_size, device=dev)
    image_padded = torch.zeros(_image_size, device=dev)
    image_cov_padded = torch.zeros(_image_size, device=dev)
    image_cov_side_padded = torch.zeros(_image_size, device=dev)
    y_start = (
        np.arange(0, padded_img_size[1] - crop_size + 1, crop_size * (1 - overlap))
    ).astype(int)
    x_start = (
        np.arange(0, padded_img_size[2] - crop_size + 1, crop_size * (1 - overlap))
    ).astype(int)
    upper_left = list(product(y_start, x_start))
    window = torch.from_numpy(window).to(dev)
    window = window[None, :, :] if naugs == 1 else window[None, :, :, None]
    # Calculate mean
    for index, crop in zip(position_index, image_crops):
        row_start, col_start = upper_left[index]
        image_padded[
        :, row_start:row_start + crop_size, col_start:col_start + crop_size
        ] += torch.multiply(window, crop)
        weights[
        :, row_start:row_start + crop_size, col_start:col_start + crop_size
        ] += window
    image_padded = torch.divide(image_padded, weights)

    # Calculate std
    for index, crop in zip(
            position_index, image_crops):
        row_start, col_start = upper_left[index]
        diff = crop - image_padded[
                      :, row_start:row_start + crop_size,
                      col_start:col_start + crop_size
                      ]
        image_cov_padded[
        :, row_start:row_start + crop_size, col_start:col_start + crop_size
        ] += torch.multiply(window, diff ** 2)
        image_cov_side_padded[
        :, row_start:row_start + crop_size, col_start:col_start + crop_size
        ] += torch.multiply(window, diff[0:1] * diff[1:2])
    image_cov_padded = torch.divide(image_cov_padded, weights)
    image_cov_side_padded = torch.divide(image_cov_side_padded, weights)

    image = image_padded[
        :,
        crop_size // 2: crop_size // 2 + img_size[1],
        crop_size // 2: crop_size // 2 + img_size[2],
    ]
    image_cov = image_cov_padded[
        :,
        crop_size // 2: crop_size // 2 + img_size[1],
        crop_size // 2: crop_size // 2 + img_size[2],
    ]
    image_cov_side = image_cov_side_padded[
         0:1,
         crop_size // 2: crop_size // 2 + img_size[1],
         crop_size // 2: crop_size // 2 + img_size[2],
    ]

    return image, image_cov, image_cov_side


def augment_image_batch(images, augmentation: int = None):
    """
    Test time augment (rotate+ flip) a batch of images
    Args:
        images: torch.tensor
            a batch of images
        augmentation: int
            list containing indices of the augmentations to apply

    Returns: torch.tensor (batch of augmented images)
    """
    # shape b,c,h,w
    img = torch.permute(images, (1, 2, 3, 0))
    img_batch = list()
    if augmentation is None:
        img_batch.append(img)
        img_batch.append(torch.rot90(img, k=1, dims=[1, 2]))
        img_batch.append(torch.rot90(img, k=2, dims=[1, 2]))
        img_batch.append(torch.rot90(img, k=3, dims=[1, 2]))
        img_batch.append(torch.flip(img, [1]))
        img_batch.append(torch.flip(img_batch[1], [1]))
        img_batch.append(torch.flip(img_batch[2], [1]))
        img_batch.append(torch.flip(img_batch[3], [1]))
    else:
        if augmentation == 0:
            img_batch.append(img)
        if augmentation == 1:
            img_batch.append(torch.rot90(img, k=1, dims=[1, 2]))
        if augmentation == 2:
            img_batch.append(torch.rot90(img, k=2, dims=[1, 2]))
        if augmentation == 3:
            img_batch.append(torch.rot90(img, k=3, dims=[1, 2]))
        if augmentation == 4:
            img_batch.append(torch.flip(img, [1]))
        if augmentation == 5:
            img_batch.append(torch.flip(torch.rot90(img, k=1, dims=[1, 2]), [1]))
        if augmentation == 6:
            img_batch.append(torch.flip(torch.rot90(img, k=2, dims=[1, 2]), [1]))
        if augmentation == 7:
            img_batch.append(torch.flip(torch.rot90(img, k=3, dims=[1, 2]), [1]))
    # shape: c,h,w, b*8
    img_batch = torch.cat(img_batch, dim=-1)
    # shape: b*8, c,h,w
    img_batch = torch.permute(img_batch, (3, 0, 1, 2))
    return img_batch


def deaugment_segmentation_batch_full(
        segmentation_batch, augmentation: int = None):
    """
    Deaugment (rotate+flip) a batch of predictions and calculate mean prediction
    Args:
        segmentation_batch: torch.tensor
              batch of augmented segmentation maps

    Returns: batch of deaugmented segmentation maps

    """
    n_augs = 8 if augmentation is None else 1
    b, c, h, w = segmentation_batch.shape
    segmentation_batch = segmentation_batch.reshape(n_augs, b // n_augs, c, h, w)
    # shape: 8,c,h,w,b
    segmentation_batch = torch.permute(segmentation_batch, (0, 2, 3, 4, 1))
    seg = list()
    if augmentation is None:
        seg.append(segmentation_batch[0])
        seg.append(rotate_segmentation_tensor(
            segmentation_batch[1].squeeze(0), k=-1, dims=[1, 2]))
        seg.append(rotate_segmentation_tensor(
            segmentation_batch[2].squeeze(0),k=-2, dims=[1, 2]))
        seg.append(rotate_segmentation_tensor(
            segmentation_batch[3].squeeze(0), k=-3, dims=[1, 2]))
        seg.append(flip_segmentation_tensor(
            segmentation_batch[4].squeeze(0), [1]))
        seg.append(rotate_segmentation_tensor(flip_segmentation_tensor(
            segmentation_batch[5].squeeze(0), [1]), k=-1, dims=[1, 2]))
        seg.append(rotate_segmentation_tensor(flip_segmentation_tensor(
            segmentation_batch[6].squeeze(0), [1]), k=-2, dims=[1, 2]))
        seg.append(rotate_segmentation_tensor(flip_segmentation_tensor(
            segmentation_batch[7].squeeze(0), [1]), k=-3, dims=[1, 2]))
    else:
        if augmentation == 0 or augmentation is None:
            seg.append(segmentation_batch[0])
        if augmentation == 1 or augmentation is None:
            seg.append(rotate_segmentation_tensor(
                segmentation_batch[0].squeeze(0), k=-1, dims=[1, 2]))
        if augmentation == 2 or augmentation is None:
            seg.append(rotate_segmentation_tensor(
                segmentation_batch[0].squeeze(0),k=-2, dims=[1, 2]))
        if augmentation == 3 or augmentation is None:
            seg.append(rotate_segmentation_tensor(
                segmentation_batch[0].squeeze(0), k=-3, dims=[1, 2]))
        if augmentation == 4 or augmentation is None:
            seg.append(flip_segmentation_tensor(
                segmentation_batch[0].squeeze(0), [1]))
        if augmentation == 5 or augmentation is None:
            seg.append(rotate_segmentation_tensor(flip_segmentation_tensor(
                segmentation_batch[0].squeeze(0), [1]), k=-1, dims=[1, 2]))
        if augmentation == 6 or augmentation is None:
            seg.append(rotate_segmentation_tensor(flip_segmentation_tensor(
                segmentation_batch[0].squeeze(0), [1]), k=-2, dims=[1, 2]))
        if augmentation == 7 or augmentation is None:
            seg.append(rotate_segmentation_tensor(flip_segmentation_tensor(
                segmentation_batch[0].squeeze(0), [1]), k=-3, dims=[1, 2]))
    # shape: 8, c, h, w, b
    seg = torch.stack(seg)
    # shape: 8,b,c,h,w
    seg = torch.permute(seg, (0, 4, 1, 2, 3))
    # shape b,c,h,w,8
    seg = torch.permute(seg, (1, 2, 3, 4, 0))

    return seg


def rotate_segmentation_tensor(segmentation, k, dims):
    offset = rotate_offset_tensor(segmentation[:2], k=k, dims=dims)
    other_classes = torch.rot90(segmentation[2:], k=k, dims=dims)
    if k % 2 and k > 0:  # if rotation of 90, 270 degree swap sigma x, sigma y as well
        sigma_y, sigma_x, seed_map = other_classes
        other_classes = torch.cat(
            [
                sigma_x[np.newaxis, ...],
                sigma_y[np.newaxis, ...],
                seed_map[np.newaxis, ...],
            ],
            dim=0,
        )

    return torch.cat([offset, other_classes], dim=0)


def flip_segmentation_tensor(segmentation, dims):
    offset = flip_offset_tensor(segmentation[:2], dims=dims)
    other_classes = torch.flip(segmentation[2:], dims=dims)
    return torch.cat([offset, other_classes], dim=0)


def deaugment_offset_batch_full(offset_batch, augmentation: int = None):
    n_augs = 8 if augmentation is None else 1
    b, c, h, w = offset_batch.shape
    offset_batch = offset_batch.reshape(n_augs, b // n_augs, c, h, w)
    # shape: 8,c,h,w,b
    offset_batch = torch.permute(offset_batch, (0, 2, 3, 4, 1))
    offset = list()
    if augmentation is None:
        offset.append(offset_batch[0].squeeze(0))
        offset.append(rotate_offset_tensor(
            offset_batch[1].squeeze(0), k=-1, dims=[1, 2]))
        offset.append(rotate_offset_tensor(
            offset_batch[2].squeeze(0), k=-2, dims=[1, 2]))
        offset.append(rotate_offset_tensor(
            offset_batch[3].squeeze(0), k=-3, dims=[1, 2]))
        offset.append(flip_offset_tensor(
            offset_batch[4].squeeze(0), [1]))
        offset.append(rotate_offset_tensor(flip_offset_tensor(
            offset_batch[5].squeeze(0), [1]), k=-1, dims=[1, 2]))
        offset.append(rotate_offset_tensor(flip_offset_tensor(
            offset_batch[6].squeeze(0), [1]), k=-2, dims=[1, 2]))
        offset.append(rotate_offset_tensor(flip_offset_tensor(
            offset_batch[7].squeeze(0), [1]), k=-3, dims=[1, 2]))
    else:
        if augmentation == 0:
            offset.append(offset_batch[0].squeeze(0))
        if augmentation == 1:
            offset.append(rotate_offset_tensor(
                offset_batch[0].squeeze(0), k=-1, dims=[1, 2]))
        if augmentation == 2:
            offset.append(rotate_offset_tensor(
                offset_batch[0].squeeze(0), k=-2, dims=[1, 2]))
        if augmentation == 3:
            offset.append(rotate_offset_tensor(
                offset_batch[0].squeeze(0), k=-3, dims=[1, 2]))
        if augmentation == 4:
            offset.append(flip_offset_tensor(
                offset_batch[0].squeeze(0), [1]))
        if augmentation == 5:
            offset.append(rotate_offset_tensor(flip_offset_tensor(
                offset_batch[0].squeeze(0), [1]), k=-1, dims=[1, 2]))
        if augmentation == 6:
            offset.append(rotate_offset_tensor(flip_offset_tensor(
                offset_batch[0].squeeze(0), [1]), k=-2, dims=[1, 2]))
        if augmentation == 7:
            offset.append(rotate_offset_tensor(flip_offset_tensor(
                offset_batch[0].squeeze(0), [1]), k=-3, dims=[1, 2]))
    # shape: 8, c, h, w, b
    offset = torch.stack(offset)
    # shape: 8,b,c,h,w
    offset = torch.permute(offset, (0, 4, 1, 2, 3))
    # shape: b,c,h,w,8
    offset = torch.permute(offset, (1, 2, 3, 4, 0))

    return offset


def rotate_offset_tensor(offset, k, dims):
    if k < 0:
        k = k % 4
    temp = torch.rot90(offset, k, dims=dims)

    if k == 2:
        temp *= -1
    if k % 2:  # 270,90 rot -> row will be col and col will be row -> flip channel
        if k == 1:
            temp[1, ...] *= -1
            temp = torch.flip(temp, dims=[0])

        if k == 3:
            temp = torch.flip(temp, dims=[0])
            temp[1, ...] *= -1
    return temp


def flip_offset_tensor(offset, dims):
    temp = torch.flip(offset, dims=dims)
    temp[0, ...] *= -1
    return temp


def create_inference_dict(
    name="2d",
    batch_size=16,
    workers=8,
):
    """
    Creates `dataset_dict` dictionary from parameters.
    Parameters
    ----------
    name: string
        "2d"
    batch_size: int
        Effective Batch-size is the product of `batch_size` and `virtual_batch_multiplier`
    workers: int
        Number of data-loader workers
    """
    if name == "2d":
        set_transforms = my_transforms.get_transform(
            [
                {
                    "name": "ToTensorFromNumpy",
                    "opts": {
                        "keys": (
                            "image_curr",
                            "image_prev",
                        ),
                        "type": (
                            torch.FloatTensor,
                            torch.FloatTensor,
                        ),
                    },
                },
            ]
        )
    else:
        raise AssertionError(f"Unknown image dimension {name}")
    dataset_dict = {
        "name": name,
        "kwargs": {
            "transform": set_transforms,
        },
        "batch_size": batch_size,
        "workers": workers,
    }

    return dataset_dict


def init_model(model_dict, configs):
    """
    Initialize the trained model
    Args:
        model_dict: dict
            parametrization of the model
        configs: dict
            configuration (model weights)

    Returns: torch.nn.Module (the initialized model)

    """
    if configs["model_class"] == "TrackERFNet":
        model = TrackERFNet(**model_dict["kwargs"])

    else:
        m_class = configs["model_class"]
        raise AssertionError(f"Unknown network type {m_class}")
    state = convert_state_dict(
        torch.load(configs["model_cktp_path"])["model_state_dict"]
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def convert_state_dict(state_dict):
    """Convert state dict if stored as data parallel model, so it can be loaded without data parallel."""
    module_chars = len("module.")
    dummy_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            dummy_state_dict[k[module_chars:]] = v
        else:
            dummy_state_dict[k] = v
    return dummy_state_dict


def smooth_prediction_full(
        data_loader, model, shifts=[]):
    """
    Predict a pair of image frames.
    Args:
        data_loader: torch.utils.data.DataLoader
            load image crops
        model: torch.nn.Module
            the trained model

    Returns:
        list (image indices), torch.tensor (batch of predicted images t), torch.tensor (batch of predicted images at t-1)
        torch.tensor (batch of predicted offsets between t->t-1)
    """
    # batch-wise prediction of masks and offsets

    all_seg_images_curr = []
    all_seg_images_prev = []
    all_offsets = []
    img_index = []

    with torch.no_grad():
        for batch in data_loader:
            img_curr_aug = augment_image_batch(batch["image_curr"]).to(device)
            img_prev_aug = augment_image_batch(batch["image_prev"]).to(device)

            if len(shifts) != 0:
                assert shifts[0] == 0
                trans = [torch.zeros_like(img_prev_aug) for s in shifts]
                for i, s in enumerate(shifts):
                    if s == 0:
                        trans[i] = img_prev_aug.clone()
                    else:
                        trans[i][:, :, :, s:] = \
                            img_prev_aug[:, :, :, :-s].clone()
                ret_t = [model(img_curr_aug, t) for t in trans]
                seg_images_curr = deaugment_segmentation_batch_full(ret_t[0][0])
                seg_images_prev = deaugment_segmentation_batch_full(ret_t[0][1])
                ret_t = [r[2] for r in ret_t]
                for i, s in enumerate(shifts):
                    if s == 0:
                        continue
                    ret_t[i][:, 1, :, :] += s / img_prev_aug.shape[3]
                offset_stack = [deaugment_offset_batch_full(r) for r in ret_t]
                offset = torch.cat(offset_stack, dim=-1)
            else:
                ret = model(img_curr_aug, img_prev_aug)
                seg_images_curr_aug, seg_images_prev_aug, offsets_aug = ret
                seg_images_curr = deaugment_segmentation_batch_full(
                    seg_images_curr_aug)
                seg_images_prev = deaugment_segmentation_batch_full(
                    seg_images_prev_aug)
                offset = deaugment_offset_batch_full(offsets_aug)
            img_index.append(batch["index"])
            all_seg_images_curr.append(seg_images_curr)
            all_seg_images_prev.append(seg_images_prev)
            all_offsets.append(offset)

        img_index = torch.cat(img_index)
        all_seg_images_curr = torch.cat(all_seg_images_curr)
        all_seg_images_prev = torch.cat(all_seg_images_prev)
        all_offsets = torch.cat(all_offsets)

    return img_index, all_seg_images_curr, all_seg_images_prev, all_offsets


def calc_mask_stack(seg, cluster, uncertainty_data, min_mask_size):
    instance_stack = []
    mus, covs, mus_old, covs_old, probs = [], [], [], [], []
    for i in range(seg.shape[-1]):
        instances = cluster_prediction(cluster, seg[..., i], min_mask_size)
        instances, mu, cov, mu_old, cov_old, prob = \
            uncertainty_data.refine_mask(instances)
        instance_stack.append(instances)
        mus.append(mu)
        covs.append(cov)
        mus_old.append(mu_old)
        covs_old.append(cov_old)
        probs.append(prob)
    return \
        torch.stack(instance_stack, dim=0), mus, covs, mus_old, covs_old, probs


def infer_image(
        img_file_t_curr,
        img_file_t_prev,
        config,
        data_config,
        model,
        shifts,
        num_seg_classes,
        num_track_classes,
        padded_img_size,
        img_size,
        scale=1.0,
):
    crops_curr_frame = generate_crops(
        os.path.join(img_file_t_curr),
        config["crop_size"],
        config["overlap"],
        scale
    )
    crops_prev_frame = generate_crops(
        os.path.join(img_file_t_prev),
        config["crop_size"],
        config["overlap"],
        scale
    )
    data_loader = torch.utils.data.DataLoader(
        InferenceDataSet(
            crops_curr_frame, crops_prev_frame, **data_config["kwargs"]
        ),
        batch_size=data_config["batch_size"] // 8,  # due to augmentation
        shuffle=False,
        drop_last=False,
        num_workers=data_config["workers"],
        pin_memory=True if device == "cuda" else False,
    )
    ret = smooth_prediction_full(data_loader, model, shifts)
    img_index, all_seg_images_curr, all_seg_images_prev, all_offsets = ret
    if scale == 1.0:
        scaled_img_size = img_size
    else:
        scaled_img_size = (int(img_size[0] * scale), int(img_size[1] * scale))
        padded_img_size = calc_padded_img_size(
            scaled_img_size, config["crop_size"], config["overlap"])[0]
    seg_prev_batch, seg_prev_c_batch, seg_prev_cs_batch = patch_crops_gpu(
        all_seg_images_prev,
        img_index,
        (num_seg_classes, *padded_img_size), (num_seg_classes, *scaled_img_size),
        config["window_func"], crop_size=config["crop_size"],
        overlap=config["overlap"])
    seg_curr_batch, seg_curr_c_batch, seg_curr_cs_batch = patch_crops_gpu(
        all_seg_images_curr,
        img_index,
        (num_seg_classes, *padded_img_size), (num_seg_classes, *scaled_img_size),
        config["window_func"], crop_size=config["crop_size"],
        overlap=config["overlap"])
    offset_batch, offset_c_batch, offset_cs_batch = patch_crops_gpu(
        all_offsets,
        img_index,
        (num_track_classes, *padded_img_size),
        (num_track_classes, *scaled_img_size),
        config["window_func"], crop_size=config["crop_size"],
        overlap=config["overlap"])

    if scale != 1.0:
        factor = 1 / scale
        seg_prev_batch[0:4] *= factor
        seg_prev_c_batch[0:4] *= factor ** 2
        seg_prev_cs_batch *= factor ** 2
        seg_curr_batch[0:4] *= factor
        seg_curr_c_batch[0:4] *= factor ** 2
        seg_curr_cs_batch *= factor ** 2
        offset_batch[0:2] *= factor
        offset_c_batch[0:2] *= factor ** 2
        offset_cs_batch *= factor ** 2

        seg_prev_batch = seg_prev_batch.permute(0, 3, 1, 2)
        seg_prev_c_batch = seg_prev_c_batch.permute(0, 3, 1, 2)
        seg_prev_cs_batch = seg_prev_cs_batch.permute(0, 3, 1, 2)
        seg_prev_batch = F.interpolate(seg_prev_batch, img_size,
                                       mode="bilinear")
        seg_prev_c_batch = F.interpolate(seg_prev_c_batch, img_size,
                                         mode="nearest")
        seg_prev_cs_batch = F.interpolate(seg_prev_cs_batch, img_size,
                                          mode="nearest")
        seg_prev_batch = seg_prev_batch.permute(0, 2, 3, 1)
        seg_prev_c_batch = seg_prev_c_batch.permute(0, 2, 3, 1)
        seg_prev_cs_batch = seg_prev_cs_batch.permute(0, 2, 3, 1)

        seg_curr_batch = seg_curr_batch.permute(0, 3, 1, 2)
        seg_curr_c_batch = seg_curr_c_batch.permute(0, 3, 1, 2)
        seg_curr_cs_batch = seg_curr_cs_batch.permute(0, 3, 1, 2)
        seg_curr_batch = F.interpolate(seg_curr_batch, img_size,
                                       mode="bilinear")
        seg_curr_c_batch = F.interpolate(seg_curr_c_batch, img_size,
                                         mode="nearest")
        seg_curr_cs_batch = F.interpolate(seg_curr_cs_batch, img_size,
                                          mode="nearest")
        seg_curr_batch = seg_curr_batch.permute(0, 2, 3, 1)
        seg_curr_c_batch = seg_curr_c_batch.permute(0, 2, 3, 1)
        seg_curr_cs_batch = seg_curr_cs_batch.permute(0, 2, 3, 1)

        offset_batch = offset_batch.permute(0, 3, 1, 2)
        offset_c_batch = offset_c_batch.permute(0, 3, 1, 2)
        offset_cs_batch = offset_cs_batch.permute(0, 3, 1, 2)
        offset_batch = F.interpolate(offset_batch, img_size,
                                    mode="bilinear")
        offset_c_batch = F.interpolate(offset_c_batch, img_size,
                                    mode="nearest", )
        offset_cs_batch = F.interpolate(offset_cs_batch, img_size,
                                    mode="nearest")
        offset_batch = offset_batch.permute(0, 2, 3, 1)
        offset_c_batch = offset_c_batch.permute(0, 2, 3, 1)
        offset_cs_batch = offset_cs_batch.permute(0, 2, 3, 1)

    return \
        seg_prev_batch, seg_prev_c_batch, seg_prev_cs_batch, \
        seg_curr_batch, seg_curr_c_batch, seg_curr_cs_batch, \
        offset_batch, offset_c_batch, offset_cs_batch


def infer_sequence(
        model,
        data_config,
        model_config,
        config,
        cluster,
        min_mask_size,
        shifts=[],
        multiscale=True,
        multisegmentation=True
):
    """
    Infer a sequence of images and store the predicted instance segmentation and the tracking offsets.
    Args:
        model: torch.nn.Module
            model to use for inference
        data_config: dict
            configuration of the data to infer
        model_config: dict
            configuration of the model to infer with
        config: dict
            paths where to store the predicted maps
        cluster: Cluster
            clustering instance
        min_mask_size: float
            threshold to remove small segmented fragments from the prediction

    """
    # generate image crops
    padded_img_size = config["padded_img_size"]
    img_size = config["img_size"]
    num_seg_classes = sum(model_config["kwargs"]["n_classes"][:-1])
    num_track_classes = model_config["kwargs"]["n_classes"][-1]
    data_dirs = dict(
        tracking_dir=os.path.join(config["res_dir"], "tracking"),
    )
    for d_path in data_dirs.values():
        if not os.path.exists(d_path):
            os.makedirs(d_path)

    img_files = {
        int(re.findall("\d+", file)[0]): os.path.join(config["image_dir"], file)
        for file in os.listdir(config["image_dir"])
    }
    time_points = sorted(img_files.keys(), reverse=True)
    lineage = dict()
    max_tracking_id = 1
    SAVE_BAYESIAN = True
    for i, t_curr_frame in enumerate(time_points[:-1]):
        summary_prev = dict()
        summary_curr = dict()
        img_file_t_curr = img_files[t_curr_frame]
        img_file_t_prev = img_files[time_points[i + 1]]

        # ### ### ###
        # A Hack for the ISBI challenge that reads in the mask files and uses them as Output
        if "/01/" in img_file_t_curr:
            mask_file_t_curr = img_file_t_curr.replace("/01/", "/01_ERR_SEG/")
        else:
            mask_file_t_curr = img_file_t_curr.replace("/02/", "/02_ERR_SEG/")
        mask_file_t_curr = os.path.join(
            os.path.dirname(mask_file_t_curr),
            "mask" + os.path.basename(mask_file_t_curr)[1:])
        if "/01/" in img_file_t_prev:
            mask_file_t_prev = img_file_t_prev.replace("/01/", "/01_ERR_SEG/")
        else:
            mask_file_t_prev = img_file_t_prev.replace("/02/", "/02_ERR_SEG/")
        mask_file_t_prev = os.path.join(
            os.path.dirname(mask_file_t_prev),
            "mask" + os.path.basename(mask_file_t_prev)[1:])

        err_mask_prev = tifffile.imread(mask_file_t_prev)
        err_mask_curr = tifffile.imread(mask_file_t_curr)
        # ### ### ###

        ret = infer_image(
            img_file_t_curr=img_file_t_curr,
            img_file_t_prev=img_file_t_prev,
            config=config,
            data_config=data_config,
            model=model,
            shifts=shifts,
            num_seg_classes=num_seg_classes,
            num_track_classes=num_track_classes,
            padded_img_size=padded_img_size,
            img_size=img_size,
            scale=1.0,
        )
        seg_prev_batch, seg_prev_c_batch, seg_prev_cs_batch, \
        seg_curr_batch, seg_curr_c_batch, seg_curr_cs_batch, \
        offset_batch, offset_c_batch, offset_cs_batch = ret
        if multiscale:
            ret = infer_image(
                img_file_t_curr=img_file_t_curr,
                img_file_t_prev=img_file_t_prev,
                config=config,
                data_config=data_config,
                model=model,
                shifts=shifts,
                num_seg_classes=num_seg_classes,
                num_track_classes=num_track_classes,
                padded_img_size=padded_img_size,
                img_size=img_size,
                scale=.65,
            )
            _seg_prev_batch, _seg_prev_c_batch, _seg_prev_cs_batch, \
                _seg_curr_batch, _seg_curr_c_batch, _seg_curr_cs_batch, \
                _offset_batch, _offset_c_batch, _offset_cs_batch = ret
            seg_prev_batch = torch.cat([seg_prev_batch, _seg_prev_batch], dim=3)
            seg_prev_c_batch = torch.cat([seg_prev_c_batch, _seg_prev_c_batch], dim=3)
            seg_prev_cs_batch = torch.cat([seg_prev_cs_batch, _seg_prev_cs_batch], dim=3)
            seg_curr_batch = torch.cat([seg_curr_batch, _seg_curr_batch], dim=3)
            seg_curr_c_batch = torch.cat([seg_curr_c_batch, _seg_curr_c_batch], dim=3)
            seg_curr_cs_batch = torch.cat([seg_curr_cs_batch, _seg_curr_cs_batch], dim=3)
            offset_batch = torch.cat([offset_batch, _offset_batch], dim=3)
            offset_c_batch = torch.cat([offset_c_batch, _offset_c_batch], dim=3)
            offset_cs_batch = torch.cat([offset_cs_batch, _offset_cs_batch], dim=3)

        _seg_prev_batch = seg_prev_batch
        _seg_prev_c_batch = seg_prev_c_batch
        _seg_prev_cs_batch = seg_prev_cs_batch
        _seg_curr_batch = seg_curr_batch
        _seg_curr_c_batch = seg_curr_c_batch
        _seg_curr_cs_batch = seg_curr_cs_batch
        _offset_batch = offset_batch
        _offset_c_batch = offset_c_batch
        _offset_cs_batch = offset_cs_batch

        if multiscale:
            n = int(offset_cs_batch.shape[-1] / 2)
            seg_prev_batch = seg_prev_batch[:, :, :, 0:8]
            seg_prev_c_batch = seg_prev_c_batch[:, :, :, 0:8]
            seg_prev_cs_batch = seg_prev_cs_batch[:, :, :, 0:8]
            seg_curr_batch = seg_curr_batch[:, :, :, 0:8]
            seg_curr_c_batch = seg_curr_c_batch[:, :, :, 0:8]
            seg_curr_cs_batch = seg_curr_cs_batch[:, :, :, 0:8]
            offset_batch = offset_batch[:, :, :, 0:n]
            offset_c_batch = offset_c_batch[:, :, :, 0:n]
            offset_cs_batch = offset_cs_batch[:, :, :, 0:n]

        # Calculate covariances
        n_augs_seg = seg_prev_batch.shape[-1]
        n_augs_offset = offset_batch.shape[-1]
        div_seg = 1 if n_augs_seg == 1 else n_augs_seg - 1
        div_off = 1 if n_augs_offset == 1 else n_augs_offset - 1
        seg_prev = seg_prev_batch.mean(dim=3, keepdim=True)
        seg_curr = seg_curr_batch.mean(dim=3, keepdim=True)
        offset = offset_batch.mean(dim=3, keepdim=True)
        seg_prev_diff = seg_prev_batch - seg_prev
        seg_prev_cov = (seg_prev_c_batch + seg_prev_diff ** 2).sum(dim=3) / div_seg

        seg_prev_c_side = (
                seg_prev_cs_batch + seg_prev_diff[0:1] * seg_prev_diff[1:2]
            ).sum(dim=3) / div_seg
        seg_prev_std = seg_prev_cov.sqrt()

        seg_curr_diff = seg_curr_batch - seg_curr
        seg_curr_cov = (seg_curr_c_batch + seg_curr_diff ** 2).sum(dim=3) / div_seg
        seg_curr_c_side = (
                seg_curr_cs_batch + seg_curr_diff[0:1] * seg_curr_diff[1:2]
            ).sum(dim=3) / div_seg
        seg_curr_std = seg_curr_cov.sqrt()

        offset_diff = offset_batch - offset
        offset_cov = (offset_c_batch + offset_diff ** 2).sum(dim=3) / div_off
        offset_c_side = (
                offset_cs_batch + offset_diff[0:1] * offset_diff[1:2]
            ).sum(dim=3) / div_off
        offset_std = offset_cov.sqrt()
        seg_prev = seg_prev[:, :, :, 0]
        seg_curr = seg_curr[:, :, :, 0]
        offset = offset[:, :, :, 0]

        # Create uncertainty data
        seg_prev_batch = _seg_prev_batch
        seg_curr_batch = _seg_curr_batch

        curr_data = infer_utils.ExtractUncertaintyData(
            cluster.grid_y, cluster.grid_x, cluster.pixel_y,
            cluster.pixel_x, seg_curr, seg_curr_std, offset, offset_std,
            offset_c_side, seg_curr_c_side)
        prev_data = infer_utils.ExtractUncertaintyData(
            cluster.grid_y, cluster.grid_x, cluster.pixel_y,
            cluster.pixel_x, seg_prev, seg_prev_std, None, None,
            None, seg_prev_c_side)

        # ### ### ### ISBI HACK
        # instances_prev_gpu = cluster_prediction(
        #     cluster, seg_prev, min_mask_size)
        # instances_prev_gpu, _, _, _, _, _ = prev_data.refine_mask(
        #     instances_prev_gpu)
        instances_prev_gpu = torch.from_numpy(
            err_mask_prev.astype(int)).to(device).to(torch.int16)
        # ### ### ###

        instances_prev = instances_prev_gpu.detach().cpu().numpy()

        if i == 0:
            # ### ### ### ISBI HACK
            # Last frame of the sequence
            # instances_curr_gpu = cluster_prediction(
            #     cluster, seg_curr, min_mask_size)
            # instances_curr_gpu, _, _, _, _, _ = curr_data.refine_mask(
            #     instances_curr_gpu)
            instances_curr_gpu = torch.from_numpy(err_mask_curr.astype(int)).to(
                device).to(torch.int16)
            # ### ### ###

            instances_curr = instances_curr_gpu.detach().cpu().numpy()
            mask_idx_curr = get_indices_pandas(instances_curr)
            max_tracking_id = max(mask_idx_curr.keys())
            lineage = {
                m_idx: [t_curr_frame, t_curr_frame, 0] for m_idx in mask_idx_curr.keys()
            }
            print(f"Save tracking mask {os.path.basename(img_file_t_curr)}")
            tifffile.imwrite(
                os.path.join(
                    data_dirs["tracking_dir"], os.path.basename(img_file_t_curr)
                ),
                instances_curr,
            )
            instances_curr = instances_curr_gpu

        else:
            instances_curr = tifffile.imread(
                os.path.join(
                    data_dirs["tracking_dir"], os.path.basename(img_file_t_curr)
                    )).astype(np.int16)
            mask_idx_curr = get_indices_pandas(instances_curr)
            instances_curr = torch.from_numpy(instances_curr).to(device)

        mask_idx_prev = get_indices_pandas(instances_prev)
        if len(mask_idx_curr.keys()) == 0:
            mask_idx_curr = dict()
        if len(mask_idx_prev.keys()) == 0:
            mask_idx_prev = dict()

        # link masks
        estimated_prev_positions = calc_previous_position_estimate(
            mask_idx_curr, offset, cluster.grid_y, cluster.grid_x,
            cluster.pixel_y, cluster.pixel_x,
        )
        # link: mask_t[new_pos] -> mask_idx_t: (mask_idx_t+1),
        links_forward = {m_idx: list() for m_idx in mask_idx_prev.keys()}
        for m_idx_curr, est_prev_position in estimated_prev_positions.items():
            m_idx, counts = np.unique(
                instances_prev[
                    est_prev_position[0].cpu(), est_prev_position[1].cpu()
                ],
                return_counts=True,)
            m_idx, counts = m_idx[m_idx != 0], counts[m_idx != 0]
            if len(counts) > 0:
                m_idx_prev = m_idx[np.argmax(counts)]
                links_forward[m_idx_prev].append(m_idx_curr)

        # generate tracking mask
        tracked_prev, lineage, max_tracking_id = link_masks(
            links_forward,
            mask_idx_prev,
            lineage,
            max_tracking_id,
            time_points[i + 1],
            instances_prev.shape,
        )

        # Generate uncertainty data
        if SAVE_BAYESIAN:
            # Current frame t
            object_states = curr_data.get_object_states(instances_curr)
            if multisegmentation:
                mask_stack_curr, mu_stack_curr, cov_stack_curr, mu_stack_old_curr, \
                    cov_stack_old_curr, probs_curr = calc_mask_stack(
                    seg_curr_batch, cluster, curr_data, min_mask_size)
                object_states = prev_data.add_image_stack_to_params(
                   object_states, mask_stack_curr, mu_stack_curr, cov_stack_curr,
                   mu_stack_old_curr, cov_stack_old_curr, probs_curr)
                object_states = curr_data.find_error_proposals(
                    object_states, instances_curr, mask_stack_curr)

            summary_curr["object_states"] = object_states
            path = os.path.join(
                data_dirs["tracking_dir"],
                os.path.basename(img_file_t_curr) + ".pkl")
            with open(path, "wb") as f:
                pickle.dump(summary_curr, f)
            path = os.path.join(
                data_dirs["tracking_dir"],
                os.path.basename(img_file_t_curr) + ".jpg")
            cv2.imwrite(path, curr_data.detection_certainty)
            # Previous frame t-1
            if i == len(time_points) - 2:
                object_states = prev_data.get_object_states(tracked_prev)
                if multisegmentation:
                    mask_stack_prev,  mu_stack_prev, cov_stack_prev, \
                        mu_stack_old_prev, cov_stack_old_prev, probs_prev = \
                        calc_mask_stack(
                            seg_prev_batch, cluster, prev_data, min_mask_size)
                    object_states = prev_data.add_image_stack_to_params(
                        object_states, mask_stack_prev, mu_stack_prev,
                        cov_stack_prev, mu_stack_old_prev, cov_stack_old_prev,
                        probs_prev)
                    object_states = curr_data.find_error_proposals(
                        object_states, tracked_prev, mask_stack_prev)
                summary_prev["object_states"] = object_states
                path = os.path.join(
                    data_dirs["tracking_dir"],
                    os.path.basename(img_file_t_prev) + ".pkl")
                with open(path, "wb") as f:
                    pickle.dump(summary_prev, f)
                if i == len(time_points) - 2:
                    path = os.path.join(
                        data_dirs["tracking_dir"],
                        os.path.basename(img_file_t_prev) + ".jpg")
                    cv2.imwrite(path, prev_data.detection_certainty)

        print(f"Save tracking mask {os.path.basename(img_file_t_prev)}")
        tifffile.imwrite(
            os.path.join(
                data_dirs["tracking_dir"], os.path.basename(img_file_t_prev)),
            tracked_prev.cpu().numpy().astype(np.uint16))

    # store lineage
    df = pd.DataFrame.from_dict(lineage, orient="index")
    df = df.reset_index().sort_values("index")
    df.to_csv(
        os.path.join(data_dirs["tracking_dir"], "res_track.txt"),
        sep=" ", index=False, header=False,
    )


def link_masks(
    links_forward, mask_idx_prev, lineage, max_tracking_id, time_point, img_shape
):
    # generate tracking mask
    track_img = torch.zeros(img_shape)
    for m_idx, successors in links_forward.items():
        if len(successors) == 0:  # new idx
            max_tracking_id += 1
            track_img[mask_idx_prev[m_idx]] = max_tracking_id
            lineage[max_tracking_id] = [time_point, time_point, 0]
        elif len(successors) == 1:  # same track
            track_id = successors[0]
            track_img[mask_idx_prev[m_idx]] = track_id
            lineage[track_id][0] = time_point  # shift t_start from t+1 -> t
        elif len(successors) == 2:  # cell division
            max_tracking_id += 1
            track_img[mask_idx_prev[m_idx]] = max_tracking_id
            lineage[max_tracking_id] = [time_point, time_point, 0]
            for s in successors:
                lineage[s][-1] = max_tracking_id  # set predecessor
        else:  # more than two cells -> treat as new track - no linking
            max_tracking_id += 1
            track_img[mask_idx_prev[m_idx]] = max_tracking_id
            lineage[max_tracking_id] = [time_point, time_point, 0]
    return track_img, lineage, max_tracking_id


def cluster_prediction(cluster, prediction, min_mask_size=None):
    """
    Cluster the predicted segmentation map into instances.
    Args:
        cluster: Cluster
            clustering instance
        prediction: torch.tensor
            predicted segmentation map
        min_mask_size: float
            threshold to remove small segmented fragments from the prediction

    """
    if min_mask_size is not None:
        segmentation = cluster.cluster_pixels(
            prediction, n_sigma=2, min_obj_size=min_mask_size
        )
    else:
        segmentation = cluster.cluster_pixels(prediction, n_sigma=2)
    if min_mask_size is not None:
        mask_ids, counts = torch.unique(segmentation, return_counts=True)
        remove_masks = mask_ids[counts < min_mask_size]
        # equivalent to array[np.isin(array, values)] = 0
        segmentation[(segmentation[..., None] == remove_masks).any(-1)] = 0
    return segmentation


def calc_previous_position_estimate(
    curr_mask_positions, offset, grid_y, grid_x, pixel_y, pixel_x
):
    """
    Calculate the estimated position of the corresponding object center in t-1.
    Args:
        curr_mask_positions: dict
            mask_id: np.array (mask positions)
        offset: np.array
            predicted offset map
        grid_y: int
            size of the grid in y direction
        grid_x: int
            size of the grid in x direction
        pixel_y: int
            size of a pixel in y direction
        pixel_x: int
            size of a pixel in x direction

    Returns: dict (estimated positions of the object center at t-1)

    """
    offset = torch.tanh(offset)
    offset[0] = torch.round((offset[0] * (grid_y - 1)) / pixel_y)
    offset[1] = torch.round((offset[1] * (grid_x - 1)) / pixel_x)
    # estimated offset: o(t+1, t)= p(t+1) - p(t)
    # estimate for previous position: p(t) = p(t+1) - o(t+1, t)
    img_shape = torch.from_numpy(
        np.array(offset.shape[1:]).reshape(-1, 1)).to(offset.device)
    if len(curr_mask_positions) == 0:
        return dict()

    estimated_prev_positions = {
        m_id: _clip_positions(
            torch.from_numpy(np.array(idx)).to(offset.device) -
            offset[..., idx[0], idx[1]].int(),
            img_shape
        )
        for m_id, idx in curr_mask_positions.items()
    }
    return estimated_prev_positions


def _clip_positions(estimated_position, img_shape):
    """Clip positions outside of the image."""
    estimated_position = estimated_position[
                         :, ~torch.any(estimated_position < 0, dim=0)]
    estimated_position = estimated_position[
        :, ~torch.any(estimated_position > img_shape - 1, dim=0)
    ]
    return estimated_position


def remove_single_frame_tracks(tracking_dir):
    """
    Remove tracks of length 1 without any predecessor.
    Args:
        tracking_dir: string
            path to the tracking directory
    """
    lineage = pd.read_csv(
        os.path.join(tracking_dir, "res_track.txt"),
        delimiter=" ",
        header=None,
        index_col=None,
        names=["track_id", "t_start", "t_end", "predecessor_id"],
    )
    res_dir = tracking_dir + "_cleaned"
    single_frame_tracks = lineage[
        (lineage["t_start"] == lineage["t_end"]) & (lineage["predecessor_id"] == 0)
    ]
    single_frame_tracks = single_frame_tracks[
        ~np.isin(single_frame_tracks["track_id"], lineage["predecessor_id"])
    ]
    tracks_by_time = single_frame_tracks.groupby("t_start", group_keys=True)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    for file in os.listdir(tracking_dir):
        if not file.endswith("tif"):
            continue
        image = tifffile.imread(os.path.join(tracking_dir, file))
        time_point = int(re.findall("\d+", file)[0])
        if time_point in tracks_by_time.groups:
            tracks_to_remove = tracks_by_time.get_group(time_point)
            for track_id in tracks_to_remove["track_id"].values:
                image[image == track_id] = 0
        tifffile.imsave(os.path.join(res_dir, file), image.astype(np.uint16))
    lineage = lineage.drop(index=single_frame_tracks.index)
    lineage.to_csv(
        os.path.join(res_dir, "res_track.txt"), sep=" ", index=False, header=False
    )


# for inference different grid (larger grid needed but the offsets where
# learned for original grid -> extend the grid while keeping the scalars
# of the original grid size and pixel size
def extend_grid(cluster, image_size):
    """
    Extend the grid to the image size. The model was trained on a small grid - now extend the grid to fit the full image size.
    Args:
        cluster: Cluster
            clustering instance
        image_size: tuple
            image size

    Returns: Cluster (clustering instance)

    """
    grid_x = max(image_size[1], cluster.grid_x)
    grid_y = max(image_size[0], cluster.grid_y)

    xm = torch.linspace(0, cluster.pixel_x, cluster.grid_x)
    ym = torch.linspace(0, cluster.pixel_y, cluster.grid_y)
    if image_size[1] > cluster.grid_x:
        xm = torch.linspace(
            0,
            cluster.pixel_x * (image_size[1] - 1) / (cluster.grid_x - 1),
            image_size[1],
        )
    if image_size[0] > cluster.grid_y:
        ym = torch.linspace(
            0,
            cluster.pixel_y * (image_size[0] - 1) / (cluster.grid_y - 1),
            image_size[0],
        )

    xm = xm.view(1, 1, -1).expand(1, grid_y, grid_x)
    ym = ym.view(1, -1, 1).expand(1, grid_y, grid_x)

    yxm = torch.cat((ym, xm), 0)
    cluster.yxm = yxm.cuda()

    return cluster


def calc_padded_img_size(img_size, crop_size, overlap):
    """Calculate the padded image size."""
    height, width = img_size
    # due to overlapping crops - additional padding needed such hat at (h, w) a window centered around (h, w)
    pad_bottom_right = (
        crop_size * (1 - overlap) - (height + crop_size) % (crop_size * (1 - overlap)),
        crop_size * (1 - overlap) - (width + crop_size) % (crop_size * (1 - overlap)),
    )
    # since we want at (0,0) a window centered around (0,0)
    pad_top_left = (crop_size // 2, crop_size // 2)
    pad_h = (int(pad_top_left[0]), int(pad_bottom_right[0]))
    pad_w = (int(pad_top_left[1]), int(pad_bottom_right[1]))
    padded_img_size = (height + sum(pad_h), width + sum(pad_w))
    return padded_img_size, (pad_h, pad_w)


def rename_to_ctc_format(data_dir, res_dir):
    """Rename tracked data to CTC naming conventions."""
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    for file in os.listdir(data_dir):
        if file.endswith("tif"):
            time_step = re.findall("\d+", file)[0]
            new_file_name = "mask" + time_step + ".tif"
        elif file.endswith("txt"):
            new_file_name = "res_track.txt"
        else:
            new_file_name = file
        shutil.copy(os.path.join(data_dir, file), os.path.join(res_dir, new_file_name))
        if file.endswith("tif"):
            img = tifffile.imread(os.path.join(res_dir, new_file_name))
            tifffile.imsave(os.path.join(res_dir, new_file_name), img.astype(np.uint16))


def foi_correction(tracking_dir, cell_type):
    """
    Adapt data sets using the field of interest definition of the CTC.
    Args:
        tracking_dir: string
            path where the segmented (and tracked) data is stored
        cell_type: string
            name of the data set

    Returns:

    """
    border_with = 0
    if cell_type in [
        "DIC-C2DH-HeLa",
        "Fluo-C2DL-Huh7",
        "Fluo-C2DL-MSC",
        "Fluo-C3DH-H157",
        "Fluo-N2DH-GOWT1",
        "Fluo-N3DH-CE",
        "Fluo-N3DH-CHO",
        "PhC-C2DH-U373",
    ]:
        border_with = 50
    if cell_type in [
        "BF-C2DL-HSC",
        "BF-C2DL-MuSC",
        "Fluo-C3DL-MDA231",
        "Fluo-N2DL-HeLa",
        "PhC-C2DL-PSC",
    ]:
        border_with = 25

    for segm_file in os.listdir(tracking_dir):
        if not segm_file.endswith(".tif"):
            continue
        segm_mask = tifffile.imread(os.path.join(tracking_dir, segm_file))
        if len(segm_mask.shape) == 2:
            segm_mask_foi = segm_mask[
                border_with : segm_mask.shape[0] - border_with,
                border_with : segm_mask.shape[1] - border_with,
            ]
        else:
            segm_mask_foi = segm_mask[
                :,
                border_with : segm_mask.shape[1] - border_with,
                border_with : segm_mask.shape[2] - border_with,
            ]
        idx_segm_mask = np.unique(segm_mask)
        idx_segm_mask = idx_segm_mask[idx_segm_mask > 0]
        idx_segm_mask_foi = np.unique(segm_mask_foi)
        idx_segm_mask_foi = idx_segm_mask_foi[idx_segm_mask_foi > 0]
        missing_masks = idx_segm_mask[~np.isin(idx_segm_mask, idx_segm_mask_foi)]
        for mask_id in missing_masks:
            segm_mask[segm_mask == mask_id] = 0
        tifffile.imsave(
            os.path.join(tracking_dir, segm_file), segm_mask.astype(np.uint16)
        )
    # remove all now missing tracks from the lineage
    remove_missing_tracks_from_lineage(tracking_dir)
    # reload all mask files -> get fragmented tracks and rename them
    edit_tracks_with_missing_masks(tracking_dir)


def remove_missing_tracks_from_lineage(tracking_dir):
    lineage_data = pd.read_csv(
        os.path.join(tracking_dir, "res_track.txt"),
        delimiter=" ",
        names=["m_id", "t_s", "t_e", "pred"],
        header=None,
        index_col=None,
    )
    tracks = {}
    for time_idx, mask_file in get_img_files(Path(tracking_dir)).items():
        mask_ids = np.unique(tifffile.imread(mask_file))
        mask_ids = mask_ids[mask_ids != 0]
        for mask_idx in mask_ids:
            if mask_idx not in tracks.keys():
                tracks[mask_idx] = []
            tracks[mask_idx].append(time_idx)
    missing_masks = set(lineage_data["m_id"].values).symmetric_difference(
        set(tracks.keys())
    )
    for mask_idx in missing_masks:
        lineage_data.loc[lineage_data["pred"] == mask_idx, "pred"] = 0
        lineage_data = lineage_data[lineage_data["m_id"] != mask_idx]
    for mask_idx, time_steps in tracks.items():
        lineage_data.loc[lineage_data["m_id"] == mask_idx, "t_s"] = min(time_steps)
        lineage_data.loc[lineage_data["m_id"] == mask_idx, "t_e"] = max(time_steps)
    lineage_data.to_csv(
        os.path.join(tracking_dir, "res_track.txt"), sep=" ", index=False, header=False
    )


def edit_tracks_with_missing_masks(tracking_dir):
    fragmented_tracks = find_tracklets(tracking_dir, "res_track.txt")
    if len(fragmented_tracks) == 0:
        return

    lineage_data = pd.read_csv(
        os.path.join(tracking_dir, "res_track.txt"),
        delimiter=" ",
        names=["m_id", "t_s", "t_e", "pred"],
        header=None,
        index_col=None,
    )
    # track_id : [t_start(gap), gap_length]
    mask_files = {
        int(re.findall("\d+", file_name)[0]): os.path.join(tracking_dir, file_name)
        for file_name in os.listdir(tracking_dir)
        if file_name.endswith(".tif")
    }

    max_track_id = lineage_data["m_id"].max()
    for track_id, t_tracklets in fragmented_tracks.items():

        tracklet_parent = 0
        for i_tracklet, (t_start, t_end) in enumerate(
            t_tracklets
        ):  # keep first tracklet unchanged as we can keep the original track id for it
            if i_tracklet == 0:
                lineage_data.loc[lineage_data["m_id"] == track_id, "t_e"] = t_end
                tracklet_parent = track_id
                continue

            max_track_id += 1
            for t in range(t_start, t_end + 1):
                mask_img = tifffile.imread(mask_files[t])
                mask_img[mask_img == track_id] = max_track_id
                tifffile.imwrite(mask_files[t], mask_img)
            lineage_data = pd.concat([lineage_data, pd.DataFrame([{
                    "m_id": max_track_id,
                    "t_s": t_start,
                    "t_e": t_end,
                    "pred": tracklet_parent,
                }])], ignore_index=True)
            # lineage_data = lineage_data.append(
            #     {
            #         "m_id": max_track_id,
            #         "t_s": t_start,
            #         "t_e": t_end,
            #         "pred": tracklet_parent,
            #     },
            #     ignore_index=True,
            # )
            tracklet_parent = max_track_id

    lineage_data.to_csv(
        os.path.join(tracking_dir, "res_track.txt"), sep=" ", index=False, header=False
    )


def find_tracklets(data_path, lineage_name="res_track.txt"):
    """
    Checks tracking results for inconsistencies between lineage file and tracking masks.
    Args:
        data_path: string or posix path to tracking directory
    """
    data_path = Path(data_path)
    lineage_data = pd.read_csv(
        (data_path / lineage_name).as_posix(),
        delimiter=" ",
        names=["m_id", "t_s", "t_e", "pred"],
        header=None,
        index_col=0,
    )
    img_files = get_img_files(Path(data_path))

    tracks = {}
    for time_point in sorted(img_files.keys()):
        tracking_masks = get_indices_pandas(tifffile.imread(img_files[time_point]))
        for mask_id in tracking_masks.index:
            if mask_id not in tracks:
                tracks[mask_id] = []
            tracks[mask_id].append(time_point)

    assert not set(tracks.keys()).symmetric_difference(
        set(lineage_data.index)
    ), f"masks in segm images: {set(tracks.keys())} and tra file {set(lineage_data.index)} not consistent"

    tracklets = {}
    for track_id, time_points in tracks.items():
        time_points = np.array(sorted(time_points))
        t_min, t_max = min(time_points), max(time_points)
        missing = t_max - t_min + 1 - len(time_points)
        if missing:
            d_t = time_points[1:] - time_points[:-1] - 1
            t_start_tracklets = [time_points[0], *time_points[1:][d_t > 0]]
            t_end_tracklets = [*time_points[:-1][d_t > 0], time_points[-1]]
            tracklets[track_id] = list(zip(t_start_tracklets, t_end_tracklets))
    return tracklets
