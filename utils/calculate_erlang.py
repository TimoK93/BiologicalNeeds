import numpy as np
import os
import argparse
import tifffile
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from scipy.stats import erlang, gamma
from datetime import datetime



def get_mask_files(mask_dir):
    mask_files = []
    for root, _, files in os.walk(mask_dir):
        for file in files:
            if file.endswith(".tif"):
                mask_files.append(os.path.join(root, file))
    return mask_files


def get_mask_ids(mask_file):
    mask = tifffile.imread(mask_file)
    mask_ids = np.unique(mask)
    return mask_ids


def load_mask_ids(mask_dir, n_jobs=1):
    mask_files = get_mask_files(mask_dir)
    if n_jobs > 1:
        with Pool(n_jobs) as p:
            mask_ids = p.map(get_mask_ids, mask_files)
    else:
        mask_ids = [get_mask_ids(mask_file) for mask_file in mask_files]
    return mask_ids


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", type=str, required=True)
    parser.add_argument("--n-jobs", type=int, default=cpu_count())
    return parser.parse_args()


def exponential_2(
        n_0,
        lambda_,
        x,
):
    """
    """
    return n_0 * 2 ** (lambda_*x)


def exponential_regression(
        x,
        y,
):
    """

    Args:
        x:
        y:

    Returns:
        lambda_, n_0

    """
    params = np.polyfit(x, np.log2(y), 1)
    lambda_ = params[0]
    n_0 = 2**params[1]
    return lambda_, n_0


def plot_erlang(mu, sigma, truth=None):
    max_x = 100
    if truth is not None:
        max_x = len(truth)
    x = np.linspace(0, max_x, max_x).astype(float)
    alpha = mu**2 / sigma**2
    beta = mu / sigma**2
    print("Erlang/Gamma values:", "alpha", alpha, "beta", beta)
    t0 = datetime.now()
    for _ in range(10000):
        erl = gamma(a=alpha, scale=1/beta)
        y = erl.cdf(x)
    print("Time:", (datetime.now() - t0) / 10000)
    erl = gamma(a=alpha, scale=1/beta)
    # y = erl.pdf(x)
    # plt.plot(x, y)
    y = erl.cdf(x)
    plt.plot(x, y)
    # if truth is not None:
    #     plt.plot(truth)
    plt.show()


def plot_exponential(n_0, lambda_, truth=None):
    max_x = 100
    if truth is not None:
        max_x = len(truth)
    x = np.linspace(0, max_x, max_x).astype(float)
    y = exponential_2(n_0, lambda_, x)
    plt.plot(x, y)
    if truth is not None:
        plt.plot(truth)
    plt.show()


if __name__ == "__main__":
    # args = get_arguments()
    # mask_ids = load_mask_ids(args.mask_dir, args.n_jobs)
    # num_ids = np.array([len(mask_id) for mask_id in mask_ids])
    # lambda_, n0 = exponential_regression(
    #     np.arange(0, len(mask_ids)),
    #     num_ids
    # )
    #
    # # Filter expected value, such that the max is the number of frames
    # if 1/lambda_ > len(mask_ids):
    #     lambda_ = 1 / len(mask_ids)
    #
    # # Estimate k of erlang distribution
    # mu = 1/lambda_
    # sigma = 1 * mu
    # print("Exponential values:", "lambda", lambda_, "n_0", n0)
    # print("Expected mean lifetime:", mu, "Expected standard deviation:", sigma)
    # lambda_ = 1 / mu
    # plot_exponential(n0, lambda_, truth=num_ids)
    # plot_erlang(mu, sigma, truth=num_ids)

    A0_Sequence01 = {
        "BF-C2DL-HSC": 25,
        "BF-C2DL-MuSC": 25,
        "DIC-C2DH-HeLa": 50,
        "Fluo-C2DL-MSC": 50,
        "Fluo-N2DH-SIM+": 0,
        "Fluo-N2DL-HeLa": 25,
        "Fluo-N2DH-GOWT1": 50,
        "PhC-C2DH-U373": 50,
        "PhC-C2DL-PSC": 25,
    }

    seqs = [
        r"train\BF-C2DL-HSC\01_RES",
        r"train\BF-C2DL-HSC\02_RES",
        r"challenge\BF-C2DL-HSC\01_RES",
        r"challenge\BF-C2DL-HSC\02_RES",
        r"train\BF-C2DL-MuSC\01_RES",
        r"train\BF-C2DL-MuSC\02_RES",
        r"challenge\BF-C2DL-MuSC\01_RES",
        r"challenge\BF-C2DL-MuSC\02_RES",
        r"train\DIC-C2DH-HeLa\01_RES",
        r"train\DIC-C2DH-HeLa\02_RES",
        r"challenge\DIC-C2DH-HeLa\01_RES",
        r"challenge\DIC-C2DH-HeLa\02_RES",
        r"train\Fluo-C2DL-MSC\01_RES",
        r"train\Fluo-C2DL-MSC\02_RES",
        r"challenge\Fluo-C2DL-MSC\01_RES",
        r"challenge\Fluo-C2DL-MSC\02_RES",
        r"train\Fluo-N2DH-SIM+\01_RES",
        r"train\Fluo-N2DH-SIM+\02_RES",
        r"challenge\Fluo-N2DH-SIM+\01_RES",
        r"challenge\Fluo-N2DH-SIM+\02_RES",
        r"train\Fluo-N2DL-HeLa\01_RES",
        r"train\Fluo-N2DL-HeLa\02_RES",
        r"challenge\Fluo-N2DL-HeLa\01_RES",
        r"challenge\Fluo-N2DL-HeLa\02_RES",
        r"train\Fluo-N2DH-GOWT1\01_RES",
        r"train\Fluo-N2DH-GOWT1\02_RES",
        r"challenge\Fluo-N2DH-GOWT1\01_RES",
        r"challenge\Fluo-N2DH-GOWT1\02_RES",
        r"train\PhC-C2DH-U373\01_RES",
        r"train\PhC-C2DH-U373\02_RES",
        r"challenge\PhC-C2DH-U373\01_RES",
        r"challenge\PhC-C2DH-U373\02_RES",
        r"train\PhC-C2DL-PSC\01_RES",
        r"train\PhC-C2DL-PSC\02_RES",
        r"challenge\PhC-C2DL-PSC\01_RES",
        r"challenge\PhC-C2DL-PSC\02_RES",
    ]
    args = get_arguments()
    for x in seqs:
        mask_ids = load_mask_ids(
            r"C:\Users\kaiser\Desktop\data\CTC\Inference\original\{}".format(x),
            args.n_jobs
        )
        num_ids = np.array([len(mask_id) for mask_id in mask_ids])
        lambda_, n0 = exponential_regression(
            np.arange(0, len(mask_ids)),
            num_ids
        )

        # Filter expected value, such that the max is the number of frames
        if 1 / lambda_ > len(mask_ids) or lambda_ <= 0:
            lambda_ = 1 / len(mask_ids)

        # Estimate k of erlang distribution
        mu = 1 / lambda_
        sigma = 1 * mu
        #print("Exponential values:", "lambda", lambda_, "n_0", n0)
        #print("Expected mean lifetime:", mu, "Expected standard deviation:", sigma)
        lambda_ = 1 / mu
        #plot_exponential(n0, lambda_, truth=num_ids)
        #plot_erlang(mu, sigma, truth=num_ids)
        print('    r"{}": {},'.format(x, int(mu)))
