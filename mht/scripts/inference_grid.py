
import os
from os.path import join, exists
import datetime
import argparse
import pandas as pd
import sys

import mht.dataloader as dl
from mht.MHT import MHTTracker
from mht.convert_to_ctc import convert_to_ctc, restore_hypothesis, \
    hypothesis_to_ndarray
from utils.postprocess import postprocess_sequence
from utils.interpolate import postprocess_sequence as interpolate_sequence
from ctc_metrics import evaluate_sequence


def infer_sequence(
        data_root,
        dest_root,
        gt_root,
        dataset_name,
        sequence_name,
        tracker_args: dict = None,
        postprocess: bool = False,
):
    # Run complete tracking and postprocessing pipeline
    in_dir = os.path.join(data_root, dataset_name, sequence_name + "_RES")
    res_dir = os.path.join(dest_root, dataset_name, sequence_name + "_RES")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    data = dl.CellTrackingChallengeSequence(
        path=data_root,
        dataset_name=dataset_name,
        sequence_name=sequence_name,
        subset="train",
    )

    kwargs = data.get_tracker_arguments()
    if tracker_args is not None:
        kwargs.update(tracker_args)

    tracker = MHTTracker(**kwargs)
    for i in range(len(data)):
        d = data.__getitem__(i, only_gaussians=True)
        tracker.step(
            z=d["z"],
            z_old=d["z_old"],
            z_id=d["z_id"],
            z_area=d["z_area"],
            z_is_at_border=d["z_is_at_border"],
            lambda_c_j=d["lambda_c_j"],
            P_D=d["P_D"],
        )
        print(f"\r(Frame {i}) {tracker}", end="")

    total_time = tracker.time_stamps[-1] - tracker.time_stamps[0]
    branch_switches = len(tracker.branch_switches)
    merged_hypotheses = sum([x2-x1 for x1, x2 in tracker.merged_hypotheses])

    # Apply postprocessing
    if postprocess:
        historical_data = tracker.get_historical_data()
        hypothesis = restore_hypothesis(historical_data)
        array = hypothesis_to_ndarray(hypothesis)
        convert_to_ctc(in_dir, res_dir, array)
        interpolation_dir = dest_root + "_interpolation"
        interpolate_sequence(
            data_root=dest_root,
            dest_root=interpolation_dir,
            dataset_name=dataset_name,
            sequence_name=sequence_name,
        )
        postprocess_sequence(
            data_root=interpolation_dir,
            dataset_name=dataset_name,
            sequence_name=sequence_name,
        )
        res_root = join(interpolation_dir, dataset_name, sequence_name + "_RES")
        gt_root = join(gt_root, dataset_name, sequence_name + "_GT")
        metrics = evaluate_sequence(res=res_root, gt=gt_root)
    else:
        metrics = dict()
    metrics["total_time"] = total_time
    metrics["branch_switches"] = branch_switches
    metrics["merged_hypotheses"] = merged_hypotheses
    return metrics


def check_if_run_is_evaluated(
        database,
        dataset_name,
        sequence_name,
        tracker_args: dict = None,
):
    if not exists(database):
        return False

    with open(database, "r") as file:
        lines = file.readlines()

    compare_dict = {
        "dataset_name": dataset_name,
        "sequence_name": sequence_name,
    }
    compare_dict.update(tracker_args)

    for line in lines:
        result_dict = eval(line)
        result_dict.pop("metrics")
        if compare_dict == result_dict:
            return True

    return False


def store_results(
        database,
        dataset_name,
        sequence_name,
        tracker_args: dict = None,
        metrics: dict = None,
):
    if not exists(database):
        with open(database, "w") as file:
            file.write("")

    with open(database, "a") as file:
        file.write(str({
            "dataset_name": dataset_name,
            "sequence_name": sequence_name,
            **tracker_args,
            "metrics": metrics,
        }) + "\n")


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("--root",)
    args.add_argument("--gt-root")
    args.add_argument("--database")
    args.add_argument("--tmp-dir")
    args.add_argument("--postprocess", action="store_true", default=True)
    args.add_argument("--shuffle", action="store_true", default=False)
    args.add_argument("--challenge")
    args.add_argument("--sequence")
    args = args.parse_args()

    root = args.root
    gt_root = args.gt_root
    tmp_dir = args.tmp_dir
    postprocess = args.postprocess
    shuffle = args.shuffle

    challenges = [
        "DIC-C2DH-HeLa",
        "BF-C2DL-HSC",
        "Fluo-N2DH-SIM+",
        "Fluo-N2DL-HeLa",
        "Fluo-C2DL-MSC",
        "Fluo-N2DH-GOWT1",
        "PhC-C2DH-U373",
        "BF-C2DL-MuSC",
        "PhC-C2DL-PSC",
    ]
    assert args.challenge in challenges, f"Invalid challenge {args.challenge}"
    challenge = args.challenge

    sequences = ["01", "02"]
    assert args.sequence in sequences, f"Invalid sequence {args.sequence}"
    sequence = args.sequence

    database = args.database + f"_{challenge}_{sequence}.txt"

    tracker_args = []

    # max_number_of_hypotheses = 250,
    # max_sampling_hypotheses = 3,
    # gating_probability = 0.01,
    # gating_distance = 10,
    # min_sampling_increment = 0.01,
    # min_object_probability = 0.1,
    # P_S = 0.9,  # 0.9
    # P_B = 0.1,  # 0.1
    # P_B_border = 0.35,
    # system_uncertainty = 0.0,

    for max_number_of_hypotheses in [1]:
        for max_sampling_hypotheses in [1]:
            for gating_probability in [0.01]:
                for gating_distance in [10]:
                    for min_sampling_increment in [0.01]:
                        for min_object_probability in [0.1]:
                            for P_S in [0.5, 0.9, 0.99]:
                                for P_B in [0.01, 0.05, 0.1, 0.3, 0.5]:
                                    for P_B_border in [0.01, 0.05, 0.1, 0.3, 0.5]:
                                        for system_uncertainty in [0.0, 0.01, 0.02, 0.05]:
                                            tracker_args.append({
                                                "max_number_of_hypotheses": max_number_of_hypotheses,
                                                "max_sampling_hypotheses": max_sampling_hypotheses,
                                                "gating_probability": gating_probability,
                                                "gating_distance": gating_distance,
                                                "min_sampling_increment": min_sampling_increment,
                                                "min_object_probability": min_object_probability,
                                                "P_S": P_S,
                                                "P_B": P_B,
                                                "P_B_border": P_B_border,
                                                "system_uncertainty": system_uncertainty,
                                            })
    if shuffle:
        import random
        random.shuffle(tracker_args)
        random.shuffle(sequences)

    for arg in tracker_args:
        if check_if_run_is_evaluated(
                database,
                challenge,
                sequence,
                tracker_args=arg,
        ):
            continue
        print("Run", challenge, sequence, arg)
        metrics = infer_sequence(
            root,
            tmp_dir,
            gt_root,
            dataset_name=challenge,
            sequence_name=sequence,
            tracker_args=arg,
            postprocess=postprocess,
        )
        store_results(
            database,
            challenge,
            sequence,
            tracker_args=arg,
            metrics=metrics,
        )







