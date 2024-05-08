import argparse
import pathlib
import os
import zipfile

import requests


CTC_URL = "http://data.celltrackingchallenge.net"

training_data_url = os.path.join(CTC_URL, "training-datasets/")
challenge_data_url = os.path.join(CTC_URL, "test-datasets/")


valid_sets = [
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
    """
    Parse command line arguments

    Returns:
        args: parsed arguments

    """
    parser = argparse.ArgumentParser(description="Download CTC data sets")
    parser.add_argument(
        "--data-path",
        type=str,
        default="",
        help="Path to the directory where the data sets will be stored",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=valid_sets,
        help="List of data sets to download",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Download training data sets",
    )
    parser.add_argument(
        "--challenge",
        action="store_true",
        help="Download challenge data sets",
    )

    args = parser.parse_args()

    if "all" in args.data_sets:
        args.data_sets = valid_sets

    for data_set in args.data_sets:
        if data_set not in valid_sets:
            raise ValueError(f"Unknown data set {data_set}")

    if not args.train and not args.challenge:
        args.train = True
        args.challenge = True

    if args.data_path == "":
        args.data_path = os.path.dirname(__file__).replace("utils", "Data")

    return args


def retrieve_ctc_data(url, save_dir):
    zip_file = os.path.join(save_dir, url.split("/")[-1])
    print("   ", url, save_dir)
    with requests.get(url, stream=True) as req:
        req.raise_for_status()
        with open(zip_file, "wb+") as file:
            for chunk in req.iter_content(chunk_size=8192):
                file.write(chunk)
    print(f"Unzip data set {os.path.basename(zip_file)}")
    with zipfile.ZipFile(zip_file) as z:
        z.extractall(save_dir)


if __name__ == "__main__":

    args = get_arguments()

    data_path = pathlib.Path(args.data_path)
    data_path.mkdir(exist_ok=True)
    (data_path / "train").mkdir(exist_ok=True)
    (data_path / "challenge").mkdir(exist_ok=True)

    if args.train:
        for data_set in args.data_sets:
            print(f"Downloading data set {data_set} ...")
            # Download training data set
            if not os.path.exists(data_path / "train" / data_set):
                dp = os.path.join(data_path, "train", data_set)
                print(f"Downloading training data set to {dp} ...")
                data_url = training_data_url + data_set + ".zip"
                retrieve_ctc_data(data_url, os.path.join(data_path, "train"))

    if args.challenge:
        for data_set in args.data_sets:
            # Download challenge data set
            if not os.path.exists(data_path / "challenge" / data_set):
                dp = os.path.join(data_path, "challenge", data_set)
                print(f"Downloading challenge data set {dp} ...")
                data_url = challenge_data_url + data_set + ".zip"
                retrieve_ctc_data(data_url, os.path.join(data_path, "challenge"))






