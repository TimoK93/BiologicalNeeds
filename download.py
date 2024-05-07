import os
from pathlib import Path


import requests
import zipfile
import os


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


sets = [    "Fluo-N2DH-SIM+",
   "Fluo-C2DL-MSC",
   "Fluo-N2DH-GOWT1",
   "PhC-C2DL-PSC",
   "BF-C2DL-HSC",
   "Fluo-N2DL-HeLa",
   "BF-C2DL-MuSC",
   "DIC-C2DH-HeLa",
   "PhC-C2DH-U373",
]

data_set = "Fluo-N2DH-SIM+"
subset = "train"  # train or challenge
sequence = "02"   # 01 or 02




ctc_data_url = "http://data.celltrackingchallenge.net"

training_data_url = os.path.join(ctc_data_url, "training-datasets/")
challenge_data_url = os.path.join(ctc_data_url, "test-datasets/")

current_path = Path.cwd()
data_path = current_path / 'ctc_raw_data'
data_path = Path("/home/kaiser/Downloads/new_datasets")
data_path.mkdir(exist_ok=True)
(data_path / "train").mkdir(exist_ok=True)
(data_path / "challenge").mkdir(exist_ok=True)


for data_set in sets:
    print(f"Downloading data set {data_set} ...")
    # Download training data set
    if not os.path.exists(data_path / "train" / data_set):
        dp = os.path.join(data_path, "train", data_set)
        print(f"Downloading training data set to {dp} ...")
        data_url = training_data_url + data_set + ".zip"
        retrieve_ctc_data(data_url, os.path.join(data_path, "train"))

    # Download challenge data set
    if not os.path.exists(data_path / "challenge" / data_set):
        dp = os.path.join(data_path, "challenge", data_set)
        print(f"Downloading challenge data set to {dp} ...")
        data_url = challenge_data_url + data_set + ".zip"
        retrieve_ctc_data(data_url, os.path.join(data_path, "challenge"))