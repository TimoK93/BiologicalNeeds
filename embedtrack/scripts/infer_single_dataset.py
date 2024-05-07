import os
import shutil
from datetime import datetime
from pathlib import Path
from time import time

from embedtrack.infer.infer_ctc_data import inference

import argparse

parser = argparse.ArgumentParser(description='Inference on a single dataset.')
parser.add_argument('--dataset', required=True, help='Dataset name')
parser.add_argument('--res-path', default=None, help='Result directory')
parser.add_argument('--model-name', default="normal", help='Dataset name')
parser.add_argument('--train', action='store_true')
parser.add_argument('--challenge', action='store_true')
parser.add_argument('--sequence', default=None, help='Dataset name')
parser.add_argument('--shifts', default="all", help="Test Time Shifts")
parser.add_argument('--multiscale', action='store_true')
parser.add_argument('--multisegmentation', action='store_true')

args = parser.parse_args()

FILE_PATH = Path(__file__)
PROJECT_PATH = os.path.join(*FILE_PATH.parts[:-4])
print(PROJECT_PATH)
if not args.train and not args.challenge:
    args.train = True
    args.challenge = True
RAW_DATA_PATHS = []
if args.train:
    RAW_DATA_PATHS.append(os.path.join(PROJECT_PATH, "Data", "train"))
if args.challenge:
    RAW_DATA_PATHS.append(os.path.join(PROJECT_PATH, "Data", "challenge"))

print(RAW_DATA_PATHS)

MODEL_PATH = os.path.join(PROJECT_PATH, "models")

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

assert args.shifts in ["all", "none"] or args.shifts.isdigit()
if args.shifts == "all":
    SHIFTS = [0, 1, 2, 4, 8]
elif args.shifts == "none":
    SHIFTS = []
else:
    assert int(args.shifts) > 0
    SHIFTS = [0, int(args.shifts)]
print(f"Shifts: {SHIFTS}")
MULTISCALE = args.multiscale
print(f"Multiscale: {MULTISCALE}")
MULTISEGMENTATION = args.multisegmentation
print(f"Multisegmentation: {MULTISEGMENTATION}")


assert args.dataset in DATA_SETS or args.dataset == "all"
if args.dataset != "all":
    DATA_SETS = [DATA_SETS[DATA_SETS.index(args.dataset)]]
SEQUENCES = ["01", "02"] if args.sequence is None else [args.sequence]

N_EPOCHS = 15
# Adam optimizer; normalize images; OneCycle LR sheduler; N epochs
if not args.model_name.endswith("_"):
    args.model_name += "_"
MODEL_NAME = args.model_name + str(N_EPOCHS)
BATCH_SIZE = 16
if args.dataset == "BF-C2DL-HSC":
    BATCH_SIZE = 16
elif args.dataset == "BF-C2DL-MuSC":
    BATCH_SIZE = 16

runtimes = {}
for raw_data_path in RAW_DATA_PATHS:
    for data_set in DATA_SETS:
        for data_id in SEQUENCES:
            res_path = args.res_path
            img_path = os.path.join(raw_data_path, data_set, data_id)
            if res_path is not None:
                split = "train" if "train" in raw_data_path else "challenge"
                res_path = os.path.join(res_path, split, data_set)
                os.makedirs(res_path, exist_ok=True)
                new_img_path = os.path.join(res_path, data_id)
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
                new_mask_path = os.path.join(res_path, data_id + "_ERR_SEG")
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

            model_dir = os.path.join(MODEL_PATH, data_set, MODEL_NAME)
            if not os.path.exists(model_dir):
                print(f"no trained model for data set {data_set}")
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

