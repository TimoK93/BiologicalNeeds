
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
    args.add_argument("--postprocess", action="store_true", default=False)
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

    MAX_NUMBER_OF_HYPOTHESES = [1, 25, 50, 75, 100, 125, 150]
    MAX_SAMPLING_HYPOTHESES = [1, 3, 5, 7]
    MAX_NUMBER_OF_HYPOTHESES = reversed([max(1, int(x)) for x in range(0, 2000, 25)])
    MAX_SAMPLING_HYPOTHESES = [max(1, int(x)) for x in range(0, 100, 2)]

    if challenge == "BF-C2DL-HSC":
        pass
    elif challenge == "BF-C2DL-MuSC":
        pass
    elif challenge == "DIC-C2DH-HeLa":
        pass
    elif challenge == "Fluo-C2DL-MSC":
        pass
    elif challenge == "Fluo-N2DH-GOWT1":
        pass
    elif challenge == "Fluo-N2DH-SIM+":
        pass
    elif challenge == "Fluo-N2DL-HeLa":
        pass
    elif challenge == "PhC-C2DH-U373":
        pass
    elif challenge == "PhC-C2DL-PSC":
        pass

    # tracker_args.append({
    #     "use_kalman_filter": True,
    # })
    # tracker_args.append({
    #     "use_kalman_filter": False,
    # })
    # tracker_args.append({
    #     "mitosis_min_length_a0": None,
    # })

    for max_number_of_hypotheses in MAX_NUMBER_OF_HYPOTHESES:
        tracker_args.append({
            "max_number_of_hypotheses": max_number_of_hypotheses,
            "max_sampling_hypotheses": 5,
        })

    for max_sampling_hypotheses in MAX_SAMPLING_HYPOTHESES:
        tracker_args.append({
            "max_number_of_hypotheses": 2000,
            "max_sampling_hypotheses": max_sampling_hypotheses,
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







