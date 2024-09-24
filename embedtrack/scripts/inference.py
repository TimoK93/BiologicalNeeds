import os
import shutil
from datetime import datetime
from pathlib import Path
from time import time
import argparse

from embedtrack.infer.infer_ctc_data import inference


ALL_DATASETS = [
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


SHIFTS = {
    "Fluo-N2DH-SIM+": 23,
    "Fluo-C2DL-MSC": 52,
    "Fluo-N2DH-GOWT1": 32,
    "PhC-C2DL-PSC": 7,
    "BF-C2DL-HSC": 10,
    "Fluo-N2DL-HeLa": 13,
    "BF-C2DL-MuSC": 19,
    "DIC-C2DH-HeLa": 57,
    "PhC-C2DH-U373": 34,
}


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Inference on a single dataset.'
    )
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--root', default="Data", help='Data root directory')
    parser.add_argument('--res-dir', default="results/embedtrack", help='Result directory')
    parser.add_argument('--model-dir', default="models", help='Model directory')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--challenge', action='store_true')
    parser.add_argument('--sequence', default=None, help='Dataset name')
    parser.add_argument('--shifts', default="default", help="Test Time Shifts")
    parser.add_argument('--multiscale', action='store_true')
    parser.add_argument('--multi-segmentation', action='store_true')
    parser.add_argument('--batch-size', default=16, help='Batch size for inference')
    parser.add_argument('--refine-segmentation', action='store_true')

    return parser.parse_args()


def refine_arguments(
        root,
        train,
        challenge,
        data,
        sequence,
        shifts,
        multiscale,
        multi_segmentation,
        refine_segmentation,
):
    # Create data paths
    raw_data_paths = []
    if not train and not challenge:
        train = True
        challenge = True
    if train:
        raw_data_paths.append(os.path.join(root, "train"))
    if challenge:
        raw_data_paths.append(os.path.join(root, "challenge"))
    print(f"Data paths: {raw_data_paths}")

    # Set shifts
    try:
        shifts = int(shifts)
    except ValueError:
        pass
    assert shifts in ["none", "default"] or isinstance(shifts, int)
    if shifts == "none":
        shifts = []
    elif shifts == "default":
        shifts = [0, SHIFTS[data]]
    else:
        assert int(shifts) > 0
        shifts = [0, int(shifts)]
    print(f"Shifts: {shifts}")

    # Select datasets
    assert data in ALL_DATASETS or data == "all"
    if data != "all":
        datasets = [ALL_DATASETS[ALL_DATASETS.index(data)]]
    else:
        datasets = ALL_DATASETS
    print(f"Datasets: {datasets}")

    # Select sequences
    sequences = ["01", "02"] if sequence is None else [sequence]
    print(f"Sequences: {sequences}")

    # Segmentation options
    print(f"Multiscale: {multiscale}")
    print(f"Multi-segmentation: {multi_segmentation}")
    print(f"Refine segmentation: {refine_segmentation}")

    return (
        raw_data_paths, datasets, sequences, shifts, multiscale,
        multi_segmentation, refine_segmentation
    )


def process(
        res_dir,
        model_dir,
        shifts,
        multiscale,
        multi_segmentation,
        batch_size,
        refine_segmentation,
        raw_data_paths,
        datasets,
        sequences,
):

    # Inference all selected datasets
    for path in raw_data_paths:
        for data in datasets:
            for seq in sequences:
                res_path = res_dir
                img_path = os.path.join(path, data, seq)
                if res_path is not None:
                    # Copy from img_path to new_img_path. Skip files that exist
                    split = "train" if "train" in path else "challenge"
                    res_path = os.path.join(res_path, split, data)
                    os.makedirs(res_path, exist_ok=True)
                    new_img_path = os.path.join(res_path, seq)
                    os.makedirs(new_img_path, exist_ok=True)
                    for root, dirs, files in os.walk(img_path):
                        for dir in dirs:
                            os.makedirs(
                                os.path.join(
                                    root.replace(img_path, new_img_path), dir
                                ),
                                exist_ok=True
                            )
                        for file in files:
                            if not os.path.exists(
                                    os.path.join(
                                        root.replace(img_path, new_img_path),
                                        file)
                            ):
                                shutil.copy(
                                    os.path.join(root, file),
                                    os.path.join(
                                        root.replace(img_path, new_img_path),
                                        file
                                    )
                                )

                    # Copy path
                    img_path = new_img_path

                # Select model
                model = os.path.join(model_dir, data)
                if not os.path.exists(model):
                    print(f"no trained model for data set {data}, {model}")
                    continue

                if os.path.exists(os.path.join(model, "best_iou_model.pth")):
                    model_path = os.path.join(model, "best_iou_model.pth")
                    config_file = os.path.join(model, "config.json")
                else:
                    timestamps_trained_models = [
                        datetime.strptime(time_stamp, "%Y-%m-%d---%H-%M-%S")
                        for time_stamp in os.listdir(model)
                    ]
                    timestamps_trained_models.sort()
                    last_model = timestamps_trained_models[-1].strftime("%Y-%m-%d---%H-%M-%S")
                    model_path = os.path.join(model, last_model, "best_iou_model.pth")
                    config_file = os.path.join(model, last_model, "config.json")

                # Perform the inference
                t_start = time()
                inference(
                    img_path,
                    model_path,
                    config_file,
                    batch_size=batch_size,
                    shifts=shifts,
                    multiscale=multiscale,
                    multisegmentation=multi_segmentation,
                    refine_segmentation=refine_segmentation,
                )
                t_end = time()

                run_time = t_end - t_start
                print(f"Inference Time {img_path}: {run_time}s")


if __name__ == "__main__":
    args = get_arguments()
    (
        raw_data_paths, datasets, sequences, shifts, multiscale,
        multi_segmentation, refine_segmentation
    ) = refine_arguments(
        args.root,
        args.train,
        args.challenge,
        args.dataset,
        args.sequence,
        args.shifts,
        args.multiscale,
        args.multi_segmentation,
        args.refine_segmentation,
    )
    process(
        model_dir=args.model_dir,
        res_dir=args.res_dir,
        shifts=shifts,
        multiscale=multiscale,
        multi_segmentation=multi_segmentation,
        batch_size=args.batch_size,
        refine_segmentation=refine_segmentation,
        raw_data_paths=raw_data_paths,
        datasets=datasets,
        sequences=sequences,
    )
