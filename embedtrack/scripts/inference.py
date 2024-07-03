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
    parser.add_argument('--shifts', default="all", help="Test Time Shifts")
    parser.add_argument('--multiscale', action='store_true')
    parser.add_argument('--multi-segmentation', action='store_true')
    parser.add_argument('--batch-size', default=16, help='Batch size for inference')
    parser.add_argument('--refine-segmentation', action='store_true')

    return parser.parse_args()


def process(
        root,
        res_dir,
        model_dir,
        train,
        challenge,
        data,
        sequence,
        shifts,
        multiscale,
        multi_segmentation,
        batch_size,
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
    assert shifts in ["all", "none"] or isinstance(shifts, int)
    if shifts == "none":
        shifts = []
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
                model_dir = os.path.join(model_dir, data)
                if not os.path.exists(model_dir):
                    print(f"no trained model for data set {data}, {model_dir}")
                    continue

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
    process(
        root=args.root,
        model_dir=args.model_dir,
        train=args.train,
        challenge=args.challenge,
        data=args.dataset,
        sequence=args.sequence,
        res_dir=args.res_dir,
        shifts=args.shifts,
        multiscale=args.multiscale,
        multi_segmentation=args.multi_segmentation,
        batch_size=args.batch_size,
        refine_segmentation=args.refine_segmentation,
    )
