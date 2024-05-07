"""
Original work Copyright 2019 Davy Neven,  KU Leuven (licensed under CC BY-NC 4.0 (https://github.com/davyneven/SpatialEmbeddings/blob/master/license.txt))
Modified work Copyright 2021 Manan Lalit, Max Planck Institute of Molecular Cell Biology and Genetics  (MIT License https://github.com/juglab/EmbedSeg/blob/main/LICENSE)
Modified work Copyright 2022 Katharina Löffler, Karlsruhe Institute of Technology (MIT License)
Modifications: processing of image pairs; augmentation of offset maps; Blur,Clahe, min max percentile augmentation
"""
import collections
import numpy as np
import torch
from torchvision.transforms import transforms as T


# from github.com/Mouseland/cellpose/omnipose/utils.py
def normalize(img, lower=0.01, upper=99.99):
    lower_perc = np.percentile(img, lower)
    upper_perc = np.percentile(img, upper)
    return np.interp(img, (lower_perc, upper_perc), (0, 1))


class ToTensorFromNumpy(object):
    def __init__(self, keys=[], type="float"):

        if isinstance(type, collections.abc.Iterable):
            assert len(keys) == len(type)

        self.keys = keys
        self.type = type

    def __call__(self, sample):

        for idx, k in enumerate(self.keys):
            t = self.type
            if isinstance(t, collections.abc.Iterable):
                t = t[idx]
            if k in sample:
                k_name = k.split("_")[0]
                if k_name == "image":  # image
                    sample[k] = torch.from_numpy(
                        normalize(sample[k].astype("float32"), lower=1.0, upper=99.0)
                    ).float()
                if k_name == "flow":
                    sample[k] = torch.from_numpy(sample[k].astype("float32")).float()
                elif (
                    k_name == "instance"
                    or k_name == "label"
                    or k_name == "center-image"
                ):
                    sample[k] = torch.from_numpy(sample[k]).short()

        return sample


def get_transform(transforms):
    transform_list = []

    for tr in transforms:
        name = tr["name"]
        opts = tr["opts"]

        transform_list.append(globals()[name](**opts))

    return T.Compose(transform_list)
