"""
Author: Katharina Löffler (2022), Karlsruhe Institute of Technology
Licensed under MIT License
Modified work Copyright 2024 Timo Kaiser, Leibniz University Hanover (MIT License)
Modifications: removed unused imports, added type hints, removed unused code
    added multiscale, shifts and multisegmentation as arguments
"""
import json
import shutil
from pathlib import Path

import numpy as np
from embedtrack.infer.inference import (
    extend_grid,
    infer_sequence,
    create_inference_dict,
    calc_padded_img_size,
    init_model,
    rename_to_ctc_format,
    device,
)
from embedtrack.utils.utils import get_img_files
from scipy.signal.windows import gaussian
import pandas as pd
import os
from embedtrack.utils.clustering import Cluster
from embedtrack.utils.create_dicts import create_model_dict
import tifffile
import torch


def inference(
        raw_data_path,
        model_path,
        config_file,
        batch_size=32,
        shifts=[],
        multiscale=False,
        multisegmentation=False,
        refine_segmentation=False,
):
    """
    Segment and track a ctc dataset using a trained EmbedTrack model.
    Args:
        raw_data_path: string
            Path to the raw images
        model_path: string
            Path to the weights of the trained model
        config_file: string
            Path to the configuration of the model
        batch_size: int
            batch size during inference
        shifts: list
            list of shifts for test time augmentation
        multiscale: bool
            whether to use multiscale inference
        multisegmentation: bool
            whether to use multisegmentation
        refine_segmentation: bool
            whether to use segmentation refinement
    """
    raw_data_path = Path(raw_data_path)
    model_path = Path(model_path)

    data_id = raw_data_path.parts[-1]
    data_set = raw_data_path.parts[-2]
    sub_set = raw_data_path.parts[-3]

    ctc_res_path = raw_data_path.parent / (data_id + "_RES")

    temp_res_path = "./temp" + data_set + "-" + data_id + "-" + sub_set
    if not os.path.exists(temp_res_path):
        os.makedirs(temp_res_path)
    else:
        shutil.rmtree(temp_res_path)

    if data_set not in model_path.as_posix():
        raise Warning(f"The model {model_path} is not named as the data set {data_set}")

    overlap = 0.5  # todo to 0.25

    with open(config_file) as file:
        train_config = json.load(file)

    model_class = train_config["model_dict"]["name"]
    crop_size = train_config["train_dict"]["crop_size"]

    image_size = tifffile.imread(
        os.path.join(raw_data_path, os.listdir(raw_data_path)[0])
    ).shape

    project_config = dict(
        image_dir=raw_data_path,
        res_dir=temp_res_path,
        model_cktp_path=model_path,
        model_class=model_class,
        grid_y=train_config["grid_dict"]["grid_y"],
        grid_x=train_config["grid_dict"]["grid_x"],
        pixel_y=train_config["grid_dict"]["pixel_y"],
        pixel_x=train_config["grid_dict"]["pixel_x"],
        overlap=overlap,
        crop_size=crop_size,  # multiple of 2
        img_size=image_size,
        padded_img_size=None,
    )
    project_config["padded_img_size"] = calc_padded_img_size(
        project_config["img_size"],
        project_config["crop_size"],
        project_config["overlap"],
    )[0]
    window_function_1d = gaussian(
        project_config["crop_size"], project_config["crop_size"] // 4
    )
    project_config["window_func"] = window_function_1d.reshape(
        -1, 1
    ) * window_function_1d.reshape(1, -1)

    dataset_dict = create_inference_dict(
        batch_size=batch_size,
    )

    # init model
    input_channels = train_config["model_dict"]["kwargs"]["input_channels"]
    n_classes = train_config["model_dict"]["kwargs"]["n_classes"]
    model_dict = create_model_dict(
        input_channels=input_channels,
        n_classes=n_classes,
    )
    model = init_model(model_dict, project_config)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    model.eval()

    # clustering
    cluster = Cluster(
        project_config["grid_y"],
        project_config["grid_x"],
        project_config["pixel_y"],
        project_config["pixel_x"],
    )
    cluster = extend_grid(cluster, image_size)
    tracking_dir = os.path.join(project_config["res_dir"], "tracking")
    infer_sequence(
        model,
        dataset_dict,
        model_dict,
        project_config,
        cluster,
        min_mask_size=train_config["train_dict"]["min_mask_size"] * 0.5,
        shifts=shifts,
        multiscale=multiscale,
        multisegmentation=multisegmentation,
        refine_segmentation=refine_segmentation,
    )
    # foi_correction(tracking_dir, data_set) todo: reactivate for standard
    # fill_empty_frames(tracking_dir) todo: reactivate for standard
    lineage = pd.read_csv(
        os.path.join(tracking_dir, "res_track.txt"), sep=" ", header=None
    )
    max_id = lineage[0].index.max()
    if max_id >= 2 ** 16 - 1:
        raise AssertionError(
            "Max Track id > 2**16 - uint16 transformation needed for ctc"
            " measure will lead to buffer overflow!"
        )
    rename_to_ctc_format(tracking_dir, ctc_res_path)
    shutil.rmtree(temp_res_path)


def fill_empty_frames(mask_dir):
    """
    Adds for each empty tracking frame the tracking result of the temporally closest frame.
    Otherwise CTC measure can yield an error.
    Args:
        tracks: a dict containing the tracking results
        time_steps: a list of time steps

    Returns: the modified tracks

    """
    time_steps = [
        (time_idx, file) for time_idx, file in get_img_files(Path(mask_dir)).items()
    ]
    time_steps.sort(key=lambda x: x[0])
    filled_time_steps = []
    empty_time_steps = []
    for time, file in time_steps:
        segm_mask = tifffile.imread(file)
        mask_ids = np.unique(segm_mask)
        mask_ids = mask_ids[mask_ids != 0]
        if len(mask_ids) > 0:
            filled_time_steps.append((time, file))
        else:
            empty_time_steps.append((time, file))

    filled_t, filled_files = list(zip(*filled_time_steps))

    lineage = pd.read_csv(
        os.path.join(mask_dir, "res_track.txt"),
        delimiter=" ",
        header=None,
        names=["cell_id", "t_start", "t_end", "predecessor"],
    )
    lineage = lineage.set_index("cell_id")
    for empty_t, empty_file in empty_time_steps:
        nearest_filled_frame = filled_files[
            np.argmin(abs(np.array(filled_t) - empty_t))
        ]
        os.remove(empty_file)
        shutil.copyfile(nearest_filled_frame, empty_file)
        new_masks = tifffile.imread(empty_file)
        mask_ids = np.unique(new_masks)
        mask_ids = mask_ids[mask_ids != 0]
        for mask_idx in mask_ids:
            lineage.loc[mask_idx, ("t_start")] = min(
                empty_t, lineage.loc[mask_idx]["t_start"]
            )
            lineage.loc[mask_idx, ("t_end")] = max(
                empty_t, lineage.loc[mask_idx]["t_end"]
            )

    lineage = lineage.reset_index().sort_values("cell_id")
    lineage.to_csv(
        os.path.join(mask_dir, "res_track.txt"), sep=" ", index=False, header=False
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    PARSER = ArgumentParser(
        description="Segmentation and Tracking using EmbedTrack Model"
    )
    PARSER.add_argument("raw_image_path", type=str)
    PARSER.add_argument("model_path", type=str)
    PARSER.add_argument("config_file_path", type=str)
    PARSER.add_argument('--multiscale', action='store_true')
    PARSER.add_argument('--shifts', default="none", help="Test Time Shifts")
    PARSER.add_argument('--multi-segmentation', action='store_true')
    PARSER.add_argument('--batch-size', default=16, help='Batch size for inference')
    PARSER.add_argument('--refine-segmentation', action='store_true')
    ARGS = PARSER.parse_args()

    assert ARGS.shifts in ["none"] or ARGS.shifts.isdigit()
    if ARGS.shifts == "none":
        SHIFTS = []
    else:
        assert int(ARGS.shifts) > 0
        SHIFTS = [0, int(ARGS.shifts)]
    print(f"Shifts: {SHIFTS}")
    print(f"Multiscale: {ARGS.multiscale}")
    print(f"Multisegmentation: {ARGS.multi_segmentation}")
    print(f"Refine Segmentation: {ARGS.refine_segmentation}")

    inference(
        raw_data_path=ARGS.raw_image_path,
        model_path=ARGS.model_path,
        config_file=ARGS.config_file_path,
        batch_size=ARGS.batch_size,
        shifts=SHIFTS,
        multiscale=ARGS.multiscale,
        multisegmentation=ARGS.multi_segmentation,
        refine_segmentation=ARGS.refine_segmentation,
    )