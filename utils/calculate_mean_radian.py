import numpy as np
import os
import argparse
import tifffile
from multiprocessing import Pool, cpu_count


def get_mask_files(mask_dir):
    mask_files = []
    for root, _, files in os.walk(mask_dir):
        for file in files:
            if file.endswith(".tif"):
                mask_files.append(os.path.join(root, file))
    return mask_files


def get_mask_areas(mask_file):
    mask = tifffile.imread(mask_file)
    mask_ids = np.unique(mask)
    mask_ids = mask_ids[mask_ids != 0]
    mask_areas = {mask_id: np.sum(mask == mask_id) for mask_id in mask_ids}
    return mask_areas


def calc_radians(areas: dict):
    radians = []
    for mask_id, area in areas.items():
        radians.append(np.sqrt(area / np.pi))
    return radians


def calculate_mean_radian(mask_dir, n_jobs=1):
    mask_files = get_mask_files(mask_dir)
    if n_jobs > 1:
        with Pool(n_jobs) as p:
            mask_areas = p.map(get_mask_areas, mask_files)
    else:
        mask_areas = [get_mask_areas(mask_file) for mask_file in mask_files]
    radians = [calc_radians(areas) for areas in mask_areas]
    radians = np.concatenate(radians)
    mean_radian = np.mean(radians)
    return mean_radian


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", type=str, required=True)
    parser.add_argument("--n-jobs", type=int, default=cpu_count())
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    mean_radian = calculate_mean_radian(args.mask_dir, args.n_jobs)
    print(mean_radian)
