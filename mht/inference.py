import os
import argparse

import dataloader as dl
from mht.MHT import MHTTracker
from mht.convert_to_ctc import convert_to_ctc, restore_hypothesis, \
    hypothesis_to_ndarray


def infer_all(
        data_root,
        dest_root,
        subset=None,
        challenge=None,
        sequence=None,
        tracker_args: dict = None,
):
    for _subset in ["train", "challenge"]:
        if subset != _subset and subset is not None:
            continue
        challenges = os.listdir(os.path.join(data_root, _subset))
        for _challenge in challenges:
            if challenge != _challenge and challenge is not None:
                continue
            for _sequence in ["01", "02"]:
                if sequence != _sequence and sequence is not None:
                    continue
                if not os.path.exists(os.path.join(
                        data_root, _subset, _challenge, _sequence+"_RES")
                ):
                    continue
                print(f"Infer {_subset} {_challenge} {_sequence}")
                infer_sequence(
                    data_root=os.path.join(data_root, _subset),
                    dest_root=os.path.join(dest_root, _subset),
                    dataset_name=_challenge,
                    sequence_name=_sequence,
                    tracker_args=tracker_args,
                )


def infer_sequence(
        data_root,
        dest_root,
        dataset_name,
        sequence_name,
        tracker_args: dict = None,
):
    in_dir = os.path.join(data_root, dataset_name, sequence_name + "_RES")
    res_dir = os.path.join(dest_root, dataset_name, sequence_name + "_RES")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    data = dl.CellTrackingChallengeSequence(
        path=data_root,
        dataset_name=dataset_name,
        sequence_name=sequence_name,
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
            z_pot_overseg=d["z_pot_overseg"],
            z_pot_underseg=d["z_pot_underseg"],
        )
        print(f"\r(After Processing Frame {i}) {tracker}", end="\n", flush=True)

    historical_data = tracker.get_historical_data()
    hypothesis = restore_hypothesis(historical_data)
    array = hypothesis_to_ndarray(hypothesis)
    convert_to_ctc(in_dir, res_dir, array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Inference of the PMBM tracker'
    )
    parser.add_argument(
        '--data-root', required=True,
        help="Path to the tracking data root"
    )
    parser.add_argument(
        '--destination-root', required=True,
        help="Path to the root of the result directory"
    )
    parser.add_argument(
        '--subset', default=None,
        help="train or challenge, if None both are evaluated"
    )
    parser.add_argument(
        '--challenge', default=None,
        help="CTC-Challenge, if None all are evaluated"
    )
    parser.add_argument(
        '--sequence', default=None,
        help="01 or 02, if None both are evaluated"
    )
    parser.add_argument(
        '--single-inference', action="store_true",
        help="Only infer a single sequence"
    )

    args, unknown = parser.parse_known_args()
    tracking_args = {}
    for arg in unknown:
        assert arg.startswith("--"), arg
        arg = arg[2:]
        key, value = arg.split("=")
        value = value.replace("'", "")
        value = value.replace('"', "")
        key = key.replace("-", "_")
        if value == "True":
            value = True
        elif value == "False":
            value = False
        elif value == "None":
            value = None
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    raise Exception(f"Unknown argument {arg}")
        tracking_args[key] = value

    if tracking_args:
        print("Tracking arguments:")
        for key, value in tracking_args.items():
            print(f"\t{key}: {value}")

    if args.single_inference:
        infer_sequence(
            data_root=args.data_root,
            dest_root=args.destination_root,
            dataset_name=args.challenge,
            sequence_name=args.sequence,
            tracker_args=tracking_args,
        )
    else:
        infer_all(
            data_root=args.data_root,
            dest_root=args.destination_root,
            subset=args.subset,
            challenge=args.challenge,
            sequence=args.sequence,
            tracker_args=tracking_args,
        )
