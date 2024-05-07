"""
Author: Katharina Löffler (2022), Karlsruhe Institute of Technology
Licensed under MIT License
"""
import os
import pandas as pd
import numpy as np
import re


def get_indices_pandas(data, background_id=0):
    """
    Extracts for each mask id its positions within the array.
    Args:
        data: a np. array with masks, where all pixels belonging to the
            same masks have the same integer value
        background_id: integer value of the background

    Returns: data frame: indices are the mask id , values the positions of the mask pixels

    """
    if data.size < 1e9:  # aim for speed at cost of high memory consumption
        masked_data = data != background_id
        flat_data = data[masked_data]  # d: data , mask attribute
        dummy_index = np.where(masked_data.ravel())[0]
        df = pd.DataFrame.from_dict({"mask_id": flat_data, "flat_index": dummy_index})
        df = df.groupby("mask_id", group_keys=True).apply(
            lambda x: np.unravel_index(x.flat_index, data.shape)
        )
    else:  # aim for lower memory consumption at cost of speed
        flat_data = data[(data != background_id)]  # d: data , mask attribute
        dummy_index = np.where((data != background_id).ravel())[0]
        mask_indices = np.unique(flat_data)
        df = {"mask_id": [], "index": []}
        data_shape = data.shape
        for mask_id in mask_indices:
            df["index"].append(
                np.unravel_index(dummy_index[flat_data == mask_id], data_shape)
            )
            df["mask_id"].append(mask_id)
        df = pd.DataFrame.from_dict(df)
        df = df.set_index("mask_id")
        df = df["index"].apply(lambda x: x)  # convert to same format as for other case
    return df


def get_img_files(img_dir, starts_with=""):
    """
    Extracts a set of tiff files from a folder.
    Args:
        img_dir: path to the image folder
        starts_with: optional string the image name needs to start with

    Returns:

    """
    img_file_pattern = re.compile(
        r"(\D*)(\d+)(\D*)\.(" + "|".join(("tif", "tiff")) + ")"
    )
    files = {
        int(img_file_pattern.match(file).groups()[1]): (img_dir / file).as_posix()
        for file in os.listdir(img_dir)
        if file.endswith(("tif", "tiff")) and file.startswith(starts_with)
    }
    return files
