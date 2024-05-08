import os
import shutil
from datetime import datetime
from pathlib import Path
from time import time

from embedtrack.infer.infer_ctc_data import inference

import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='Inference on a single dataset.')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--res-path', default="results/embedtrack", help='Result directory')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--challenge', action='store_true')
    parser.add_argument('--sequence', default=None, help='Dataset name')
    parser.add_argument('--shifts', default="all", help="Test Time Shifts")
    parser.add_argument('--multiscale', action='store_true')
    parser.add_argument('--multi-segmentation', action='store_true')
    return parser.parse_args()


def process(
        train,
        challenge,
        dataset,
        sequence,
        res_path,
        shifts,
        multiscale,
        multi_segmentation,
):
    FILE_PATH = Path(__file__)
    PROJECT_PATH = os.path.join(*FILE_PATH.parts[:-4])
    CWD = os.getcwd()
    print(PROJECT_PATH)
    if not train and not challenge:
        train = True
        challenge = True
    RAW_DATA_PATHS = []
    if train:
        RAW_DATA_PATHS.append(os.path.join(CWD, "Data", "train"))
    if challenge:
        RAW_DATA_PATHS.append(os.path.join(CWD, "Data", "challenge"))

    print(RAW_DATA_PATHS)

    MODEL_PATH = os.path.join(CWD, "models")

    DATA_SETS = [
        "Fluo-N2DH-SIM+",
        "Fluo-C2DL-MSC",
        "Fluo-N2DH-GOWT1",
        "PhC-C2DL-PSC",
        "BF-C2DL-HSC",
        "Fluo-N2DL-HeLa",
        "BF-C2DL-MuSC",
        "DIC-C2DH-HeLa",
        "PhC-C2DH-U373",
    ]

    assert shifts in ["all", "none"] or isinstance(shifts, int)
    if shifts == "all":
        SHIFTS = [0, 1, 2, 4, 8]
    elif shifts == "none":
        SHIFTS = []
    else:
        assert int(shifts) > 0
        SHIFTS = [0, int(shifts)]
    print(f"Shifts: {SHIFTS}")
    MULTISCALE = multiscale
    print(f"Multiscale: {MULTISCALE}")
    MULTISEGMENTATION = multi_segmentation
    print(f"Multisegmentation: {MULTISEGMENTATION}")

    assert dataset in DATA_SETS or dataset == "all"
    if dataset != "all":
        DATA_SETS = [DATA_SETS[DATA_SETS.index(dataset)]]
    SEQUENCES = ["01", "02"] if sequence is None else [sequence]

    BATCH_SIZE = 16

    for raw_data_path in RAW_DATA_PATHS:
        for data_set in DATA_SETS:
            for data_id in SEQUENCES:
                _res_path = res_path
                img_path = os.path.join(raw_data_path, data_set, data_id)
                if _res_path is not None:
                    split = "train" if "train" in raw_data_path else "challenge"
                    _res_path = os.path.join(_res_path, split, data_set)
                    os.makedirs(_res_path, exist_ok=True)
                    new_img_path = os.path.join(_res_path, data_id)
                    os.makedirs(new_img_path, exist_ok=True)
                    # Copy the tree from img_path to new_img_path. Skip files that
                    # do exist
                    for root, dirs, files in os.walk(img_path):
                        for dir in dirs:
                            os.makedirs(
                                os.path.join(root.replace(img_path, new_img_path), dir),
                                exist_ok=True)
                        for file in files:
                            if not os.path.exists(
                                    os.path.join(root.replace(img_path, new_img_path),
                                                 file)):
                                shutil.copy(os.path.join(root, file),
                                            os.path.join(
                                                root.replace(img_path, new_img_path),
                                                file))

                    # ### ### ### ISBI CHALLENGE HACK
                    # Copy masks if existing
                    mask_path = os.path.join(img_path + "_ERR_SEG")
                    new_mask_path = os.path.join(_res_path, data_id + "_ERR_SEG")
                    if os.path.exists(mask_path):
                        os.makedirs(new_mask_path, exist_ok=True)
                        for root, dirs, files in os.walk(mask_path):
                            for dir in dirs:
                                os.makedirs(
                                    os.path.join(
                                        root.replace(mask_path, new_mask_path), dir),
                                    exist_ok=True)
                            for file in files:
                                if not os.path.exists(
                                        os.path.join(
                                            root.replace(mask_path, new_mask_path),
                                            file)):
                                    shutil.copy(os.path.join(root, file),
                                                os.path.join(
                                                    root.replace(mask_path,
                                                                 new_mask_path),
                                                    file))
                    # ### ### ###

                    # Copy path
                    img_path = new_img_path

                model_dir = os.path.join(MODEL_PATH, data_set)
                if not os.path.exists(model_dir):
                    print(f"no trained model for data set {data_set}, {model_dir}")
                    continue

                # time stamps
                if os.path.exists(os.path.join(model_dir, "best_iou_model.pth")):
                    model_path = os.path.join(model_dir, "best_iou_model.pth")
                    config_file = os.path.join(model_dir, "config.json")
                else:
                    timestamps_trained_models = [
                        datetime.strptime(time_stamp, "%Y-%m-%d---%H-%M-%S")
                        for time_stamp in os.listdir(model_dir)
                    ]
                    timestamps_trained_models.sort()
                    last_model = timestamps_trained_models[-1].strftime("%Y-%m-%d---%H-%M-%S")
                    model_path = os.path.join(model_dir, last_model, "best_iou_model.pth")
                    config_file = os.path.join(model_dir, last_model, "config.json")
                t_start = time()
                inference(img_path, model_path, config_file,
                          batch_size=BATCH_SIZE, shifts=SHIFTS,
                          multiscale=MULTISCALE,
                          multisegmentation=MULTISEGMENTATION)
                t_end = time()

                run_time = t_end - t_start
                print(f"Inference Time {img_path}: {run_time}s")


if __name__ == "__main__":
    args = get_arguments()
    process(
        train=args.train,
        challenge=args.challenge,
        dataset=args.dataset,
        sequence=args.sequence,
        res_path=args.res_path,
        shifts=args.shifts,
        multiscale=args.multiscale,
        multi_segmentation=args.multi_segmentation,
    )



