import numpy as np
import os
import argparse
import pickle
import tifffile
from multiprocessing import Pool, cpu_count


def get_pkl_files(pkl_dir):
    pkl_files = []
    for root, _, files in os.walk(pkl_dir):
        for file in files:
            if file.endswith(".tif.pkl"):
                pkl_files.append(os.path.join(root, file))
    return pkl_files


def get_image_size(pkl_dir):
    for root, _, files in os.walk(pkl_dir):
        for file in files:
            if file.endswith(".tif"):
                tifffile.imread(os.path.join(root, file))
                h, w = tifffile.imread(os.path.join(root, file)).shape
                return h, w
    assert "No image found in the mask directory"


def get_uncertainties(pkl_file):
    pkl_file = pkl_file
    data = pickle.load(open(pkl_file, "rb"))
    uncertainties = data["object_states"]["old_cov"]
    uncertainties = np.array(uncertainties)
    uncertainties = np.maximum(uncertainties[:, 0, 0], uncertainties[:, 1, 1])
    uncertainties = np.sqrt(uncertainties)
    return uncertainties


def calculate_mean_uncertainty(mask_dir, h, w, n_jobs=1):
    pkl_files = get_pkl_files(mask_dir)
    if n_jobs > 1:
        with Pool(n_jobs) as p:
            uncertainties = p.map(get_uncertainties, pkl_files)
    else:
        uncertainties = [get_uncertainties(mask_file) for mask_file in pkl_files]
    values = np.concatenate(uncertainties)
    mean_value = np.mean(values) * (h + w) / 2
    return mean_value


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=cpu_count())
    parser.add_argument("--result-dirs", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    results = dict()
    for dataset in args.datasets:
        dirs = dict()
        for result_dir in args.result_dirs:
            for subset in ["train"]:
                mean_uncertainty = 0
                for seq in ["01", "02"]:
                    pkl_dir = os.path.join(result_dir, subset, dataset, f"{seq}_RES")
                    h, w = get_image_size(pkl_dir)
                    mean_uncertainty += 0.5 * calculate_mean_uncertainty(
                        pkl_dir, h, w, args.n_jobs)
                dirs[result_dir] = mean_uncertainty
        results[dataset] = dirs
        print(f"Mean uncertainty for {dataset}: {dirs}")

    for dataset, dirs in results.items():
        print(f"& {dataset}", end="")
        start_uncertainty = None
        for result_dir, mean_uncertainty in dirs.items():
            if start_uncertainty is None:
                start_uncertainty = mean_uncertainty
                print(f" & {mean_uncertainty:.2f}", end="")
            else:
                mean_uncertainty = mean_uncertainty / start_uncertainty
                print(r" & $\times" + f"{mean_uncertainty:.2f}$", end="")
        print(r"\\")
