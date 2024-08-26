
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
        try:
            metrics = evaluate_sequence(res=res_root, gt=gt_root)
        except Exception as e:
            print(e)
            metrics = dict()
    else:
        metrics = dict()
    metrics["total_time"] = total_time
    metrics["branch_switches"] = branch_switches
    metrics["merged_hypotheses"] = merged_hypotheses
    return metrics


def check_if_run_is_evaluated(
        database,
        dataset_name,
        tracker_args: dict = None,
):
    if not exists(database):
        return False

    with open(database, "r") as file:
        lines = file.readlines()

    compare_dict = {
        "dataset_name": dataset_name,
    }
    compare_dict.update(tracker_args)

    for line in lines:
        try:
            result_dict = eval(line)
            result_dict.pop("metrics")
            if compare_dict == result_dict:
                return True
        except ValueError as e:
            pass
    return False


def store_results(
        database,
        dataset_name,
        tracker_args: dict = None,
        metrics: dict = None,
):
    if not exists(database):
        with open(database, "w") as file:
            file.write("")

    with open(database, "a") as file:
        file.write(str({
            "dataset_name": dataset_name,
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
    args.add_argument("--sequence", default="all")
    args = args.parse_args()

    root = args.root
    gt_root = args.gt_root
    tmp_dir = args.tmp_dir
    postprocess = args.postprocess
    shuffle = args.shuffle

    challenges = [
        "BF-C2DL-HSC",
        "BF-C2DL-MuSC",
        "DIC-C2DH-HeLa",
        "Fluo-C2DL-MSC",
        "Fluo-N2DH-GOWT1",
        "Fluo-N2DH-SIM+",
        "Fluo-N2DL-HeLa",
        "PhC-C2DH-U373",
        "PhC-C2DL-PSC",
    ]
    assert args.challenge in challenges, f"Invalid challenge {args.challenge}"
    challenge = args.challenge

    sequences = ["01", "02"]
    assert args.sequence in sequences or args.sequence == "all", f"Invalid sequence {args.sequence}"
    if args.sequence != "all":
        sequences = [args.sequence]
        database = os.path.join(args.database, f"{challenge}_{args.sequence}.txt")
    else:
        database = os.path.join(args.database, f"{challenge}.txt")
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

    MAX_NUMBER_OF_HYPOTHESES = [1, 50, 150]
    MAX_SAMPLING_HYPOTHESES = [1, 3, 5]
    GATING_PROBABILITY = [0.01]
    GATING_DISTANCE = [10]
    MIN_SAMPLING_INCREMENT = [0.01]
    MIN_OBJECT_PROBABILITY = [0.1]
    P_S = [0.5, 0.9, 0.99]
    P_B = [0.01, 0.1, 0.3, 0.5]
    P_B_BORDER = [0.01, 0.1, 0.3, 0.5]
    SYSTEM_UNCERTAINTY = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]

    if challenge == "BF-C2DL-HSC":
        MAX_NUMBER_OF_HYPOTHESES = [1, 50, 150, 250, 500, 1000]
        MAX_SAMPLING_HYPOTHESES = [3, 5, 7]
        P_S = [0.99]
        P_B_BORDER = None
        P_B = [0.01]
        SYSTEM_UNCERTAINTY = [0.02]
        GATING_PROBABILITY = [0.01]
        GATING_DISTANCE = [10]
        MIN_SAMPLING_INCREMENT = [0.01]
        MIN_OBJECT_PROBABILITY = [0.01]
    elif challenge == "BF-C2DL-MuSC":
        MAX_NUMBER_OF_HYPOTHESES = [250]
        MAX_SAMPLING_HYPOTHESES = [5]
        P_S = [0.9]
        P_B_BORDER = [0.01]
        P_B = [0.01]
        SYSTEM_UNCERTAINTY = [0.02]
        GATING_PROBABILITY = [0.01,]
        GATING_DISTANCE = [10,]
        MIN_SAMPLING_INCREMENT = [0.01]
        MIN_OBJECT_PROBABILITY = [0.1,]
    elif challenge == "DIC-C2DH-HeLa":
        MAX_NUMBER_OF_HYPOTHESES = [150]
        MAX_SAMPLING_HYPOTHESES = [7]
        P_S = [0.5]
        P_B_BORDER = [0.1]
        P_B = [0.1]
        SYSTEM_UNCERTAINTY = [0.01]
        GATING_PROBABILITY = [0.01]
        GATING_DISTANCE = [10]
        MIN_SAMPLING_INCREMENT = [0.01]
        MIN_OBJECT_PROBABILITY = [0.1]
    elif challenge == "Fluo-C2DL-MSC":
        MAX_NUMBER_OF_HYPOTHESES = [150]
        MAX_SAMPLING_HYPOTHESES = [7]
        P_S = [0.5]
        P_B_BORDER = [0.5]
        P_B = [0.4]
        SYSTEM_UNCERTAINTY = [0.05]
        GATING_PROBABILITY = [0.01]
        GATING_DISTANCE = [10]
        MIN_SAMPLING_INCREMENT = [0.01]
        MIN_OBJECT_PROBABILITY = [0.1]
    elif challenge == "Fluo-N2DH-GOWT1":
        MAX_NUMBER_OF_HYPOTHESES = [150]
        MAX_SAMPLING_HYPOTHESES = [7]
        P_S = [0.99]
        P_B_BORDER = [0.5]
        P_B = [0.3]
        SYSTEM_UNCERTAINTY = [0.02]
        GATING_PROBABILITY = [0.01]
        GATING_DISTANCE = [10]
        MIN_SAMPLING_INCREMENT = [0.01]
        MIN_OBJECT_PROBABILITY = [0.1]
    elif challenge == "Fluo-N2DH-SIM+":
        MAX_NUMBER_OF_HYPOTHESES = [150]
        MAX_SAMPLING_HYPOTHESES = [7]
        P_S = [0.99]
        P_B_BORDER = [0.5]
        P_B = [0.5]
        SYSTEM_UNCERTAINTY = [0.05]
        GATING_PROBABILITY = [0.01]
        GATING_DISTANCE = [10]
        MIN_SAMPLING_INCREMENT = [0.01]
        MIN_OBJECT_PROBABILITY = [0.1]
    elif challenge == "Fluo-N2DL-HeLa":
        MAX_NUMBER_OF_HYPOTHESES = [1, 50, 100, 150, 250, 500]
        MAX_SAMPLING_HYPOTHESES = [1, 5, 7]
        P_S = [0.9]
        P_B_BORDER = [0.5]
        P_B = [0.1]
        SYSTEM_UNCERTAINTY = [0.01]
        GATING_PROBABILITY = [0.01]
        GATING_DISTANCE = [10]
        MIN_SAMPLING_INCREMENT = [0.01]
        MIN_OBJECT_PROBABILITY = [0.1]
    elif challenge == "PhC-C2DH-U373":
        MAX_NUMBER_OF_HYPOTHESES = [150]
        MAX_SAMPLING_HYPOTHESES = [7]
        P_S = [0.9]
        P_B_BORDER = None
        P_B = [0.1]
        SYSTEM_UNCERTAINTY = [0.01]
        GATING_PROBABILITY = [0.01]
        GATING_DISTANCE = [10]
        MIN_SAMPLING_INCREMENT = [0.01]
        MIN_OBJECT_PROBABILITY = [0.1]
    elif challenge == "PhC-C2DL-PSC":
        MAX_NUMBER_OF_HYPOTHESES = [1, 50, 100, 150, 250, 500]
        MAX_SAMPLING_HYPOTHESES = [1, 5, 7]
        P_S = [0.99]
        P_B = [0.01]
        P_B_BORDER = None
        SYSTEM_UNCERTAINTY = [0.0]
        GATING_PROBABILITY = [0.01]
        GATING_DISTANCE = [10]
        MIN_SAMPLING_INCREMENT = [0.01]
        MIN_OBJECT_PROBABILITY = [0.01]


    for max_number_of_hypotheses in MAX_NUMBER_OF_HYPOTHESES:
        for max_sampling_hypotheses in MAX_SAMPLING_HYPOTHESES:
            for gating_probability in GATING_PROBABILITY:
                for gating_distance in GATING_DISTANCE:
                    for min_sampling_increment in MIN_SAMPLING_INCREMENT:
                        for min_object_probability in MIN_OBJECT_PROBABILITY:
                            for p_s in P_S:
                                for p_b in P_B:
                                    P_B_SET = P_B_BORDER if P_B_BORDER is not None else [p_b]
                                    for p_b_border in P_B_SET:
                                        for system_uncertainty in SYSTEM_UNCERTAINTY:
                                            tracker_args.append({
                                                "max_number_of_hypotheses": max_number_of_hypotheses,
                                                "max_sampling_hypotheses": max_sampling_hypotheses,
                                                "gating_probability": gating_probability,
                                                "gating_distance": gating_distance,
                                                "min_sampling_increment": min_sampling_increment,
                                                "min_object_probability": min_object_probability,
                                                "P_S": p_s,
                                                "P_B": p_b,
                                                "P_B_border": p_b_border,
                                                "system_uncertainty": system_uncertainty,
                                            })

    if shuffle:
        import random
        random.shuffle(tracker_args)

    for i, arg in enumerate(tracker_args):
        if check_if_run_is_evaluated(
                database,
                challenge,
                tracker_args=arg,
        ):
            continue
        print(f"({i}/{len(tracker_args)})  Run", challenge, arg)
        metrics = list()
        for sequence in sequences:
            metrics.append(infer_sequence(
                root,
                tmp_dir,
                gt_root,
                dataset_name=challenge,
                sequence_name=sequence,
                tracker_args=arg,
                postprocess=postprocess,
            ))
        # average the metrics
        _m = dict()
        for key in metrics[0]:
            num_values = 0
            value = None
            for m in metrics:
                if key in m:
                    if m[key] is not None:
                        num_values += 1
                        if value is None:
                            value = m[key]
                        else:
                            value += m[key]
            if num_values > 0:
                _m[key] = value / num_values
        metrics = _m

        store_results(
            database,
            challenge,
            tracker_args=arg,
            metrics=metrics,
        )







