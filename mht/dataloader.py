import numpy as np
import tifffile as tiff
import pickle
from os import listdir
from os.path import join, basename, dirname
import cv2

from mht.utils import Gaussians, PoissonPointProcess, gaussian_pdf

ESTIMATED_MUS = {
    r"train\BF-C2DL-HSC\01_RES": 921,
    r"train\BF-C2DL-HSC\02_RES": 339,
    r"challenge\BF-C2DL-HSC\01_RES": 305,
    r"challenge\BF-C2DL-HSC\02_RES": 262,
    r"train\BF-C2DL-MuSC\01_RES": 820,
    r"train\BF-C2DL-MuSC\02_RES": 326,
    r"challenge\BF-C2DL-MuSC\01_RES": 340,
    r"challenge\BF-C2DL-MuSC\02_RES": 382,
    r"train\DIC-C2DH-HeLa\01_RES": 84,
    r"train\DIC-C2DH-HeLa\02_RES": 84,
    r"challenge\DIC-C2DH-HeLa\01_RES": 115,
    r"challenge\DIC-C2DH-HeLa\02_RES": 115,
    r"train\Fluo-C2DL-MSC\01_RES": 48,
    r"train\Fluo-C2DL-MSC\02_RES": 48,
    r"challenge\Fluo-C2DL-MSC\01_RES": 48,
    r"challenge\Fluo-C2DL-MSC\02_RES": 48,
    r"train\Fluo-N2DH-SIM+\01_RES": 65,
    r"train\Fluo-N2DH-SIM+\02_RES": 59,
    r"challenge\Fluo-N2DH-SIM+\01_RES": 110,
    r"challenge\Fluo-N2DH-SIM+\02_RES": 55,
    r"train\Fluo-N2DL-HeLa\01_RES": 53,
    r"train\Fluo-N2DL-HeLa\02_RES": 64,
    r"challenge\Fluo-N2DL-HeLa\01_RES": 52,
    r"challenge\Fluo-N2DL-HeLa\02_RES": 76,
    r"train\Fluo-N2DH-GOWT1\01_RES": 92,
    r"train\Fluo-N2DH-GOWT1\02_RES": 92,
    r"challenge\Fluo-N2DH-GOWT1\01_RES": 92,
    r"challenge\Fluo-N2DH-GOWT1\02_RES": 92,
    r"train\PhC-C2DH-U373\01_RES": 115,
    r"train\PhC-C2DH-U373\02_RES": 115,
    r"challenge\PhC-C2DH-U373\01_RES": 115,
    r"challenge\PhC-C2DH-U373\02_RES": 115,
    r"train\PhC-C2DL-PSC\01_RES": 88,
    r"train\PhC-C2DL-PSC\02_RES": 98,
    r"challenge\PhC-C2DL-PSC\01_RES": 89,
    r"challenge\PhC-C2DL-PSC\02_RES": 88,
}


def get_img_files(img_path: str):
    """
    Get all images in a folder and sort them.
    Args:
        img_path: str
            Path to the folder containing the images
    Returns:
        List of paths to the images
    """
    img_files = [
        join(img_path, x) for x in listdir(img_path) if
        x.endswith(".tif") or x.endswith(".tiff")
    ]
    img_files.sort()
    return img_files


def get_mask_files(mask_path: str):
    """
    Get all masks in a folder and sort them.
    Args:
        mask_path: str
            Path to the folder containing the masks
    Returns:
        List of paths to the masks
    """
    return get_img_files(mask_path)


def get_gaussians(img_path: str):
    """
    Get all predictions in a folder and sort them.
    Args:
        img_path: str
            Path to the folder containing the predictions
    Returns:
        List of paths to the predictions
    """
    files = [join(img_path, x) for x in listdir(img_path) if x.endswith(".pkl")]
    files.sort()
    return files


class CellTrackingChallengeSequence:

    PARAMETER_SETTINGS = {
        "BF-C2DL-HSC": {
            "mitosis_min_length_a0": 500,
            "P_B": 0.01, "P_S": 0.99,
            "max_sampling_hypotheses": 7,
            "system_uncertainty": 0.02,
        },
        "BF-C2DL-MuSC": {
            "mitosis_min_length_a0": 400,
            "P_B": 0.3, "P_S": 0.99,
            "max_sampling_hypotheses": 7,
            "split_likelihood": .05,
            "segmentation_errors": True,
        },
        "DIC-C2DH-HeLa": {
            "mitosis_min_length_a0": 80,
        },
        "Fluo-C2DL-MSC": {
            "mitosis_min_length_a0": 20,
        },
        "Fluo-N2DH-SIM+": {
            "mitosis_min_length_a0": 30,
            "split_likelihood": 0.1,
        },
        "Fluo-N2DL-HeLa": {
            "mitosis_min_length_a0": 80,
            "split_likelihood": 0.1,
        },
        "Fluo-N2DH-GOWT1": {
            "mitosis_min_length_a0": 20,
            "P_B": 0.50,
        },
        "PhC-C2DH-U373": {
            "mitosis_min_length_a0": 50,
            "P_B": 0.5, "P_S": 0.5,
        },
        "PhC-C2DL-PSC": {
            "mitosis_min_length_a0": 150,
            "max_sampling_hypotheses": 7,
        },
    }

    BORDER_WIDTH = {
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

    def get_tracker_arguments(self):
        ret = self.PARAMETER_SETTINGS[self.dataset_name]
        return ret

    def __init__(
            self,
            path: str,
            dataset_name: str,
            sequence_name: str,
            multiprocessing: bool = True
    ):
        self.dataset_name = dataset_name
        self.dataset_path = join(path, dataset_name)
        self.img_files = get_img_files(join(self.dataset_path, sequence_name))
        self.mask_files = get_mask_files(
            join(self.dataset_path, sequence_name + "_RES"))
        self.prediction_files = get_gaussians(
            join(self.dataset_path, sequence_name + "_RES"))
        self.multiprocessing = multiprocessing

    def __len__(self):
        return len(self.prediction_files)

    def __getitem__(self, idx, only_gaussians=True):
        # Load and verify data
        with open(self.prediction_files[idx], "rb") as file:
            ret = pickle.load(file)

        state = ret["object_states"]
        assert ~np.isnan(state["x"]).any()
        assert ~np.isnan(state["y"]).any()
        assert ~np.isnan(state["cov"]).any()
        assert ~np.isnan(state["old_x"]).any()
        assert ~np.isnan(state["old_y"]).any()
        assert ~np.isnan(state["old_cov"]).any()
        assert ~np.isnan(state["id"]).any()
        assert ~np.isnan(state["certainty_probability_of_confusion"]).any()

        # Create inputs
        if len(state["x"]) > 0:
            z = Gaussians(
                mu=np.stack([state["x"], state["y"]], axis=1),
                covariances=np.asarray(state["cov"])
            )
            z_old = Gaussians(
                mu=np.stack([state["old_x"], state["old_y"]], axis=1),
                covariances=np.asarray(state["old_cov"])
            )
        else:
            z = Gaussians(mu=np.empty((0, 2)), covariances=np.empty((0, 2, 2)))
            z_old = Gaussians(
                mu=np.empty((0, 2)), covariances=np.empty((0, 2, 2))
            )
        z_id = np.asarray(state["id"], dtype=int)
        z_area = np.asarray(state["area"], dtype=int)
        lambda_c_j = 1 - np.asarray(state["certainty_probability_of_confusion"])

        certainty_img_path = join(
            dirname(self.mask_files[idx]),
            basename(self.img_files[idx])+'.jpg'
        )
        P_D = cv2.imread(certainty_img_path, cv2.IMREAD_GRAYSCALE)
        P_D = P_D.astype(np.float32) / 255
        P_D = (1 + P_D) / 2
        P_D = np.clip(P_D, 0.02, 0.8)

        dw = self.BORDER_WIDTH[self.dataset_name] / P_D.shape[1]
        dh = self.BORDER_WIDTH[self.dataset_name] / P_D.shape[0]
        z_is_at_border = np.zeros(len(z_id), dtype=bool)
        z_is_at_border = np.logical_or(z_is_at_border, z.mu[:, 0] < dw)
        z_is_at_border = np.logical_or(z_is_at_border, z.mu[:, 0] > 1 - dw)
        z_is_at_border = np.logical_or(z_is_at_border, z.mu[:, 1] < dh)
        z_is_at_border = np.logical_or(z_is_at_border, z.mu[:, 1] > 1 - dh)

        z_pot_overseg = list()
        z_pot_underseg = list()
        if "img_stack_meta_info" in state:
            img_stack_meta_info = state["img_stack_meta_info"]
            error_proposals = state["error_proposals"]

            proposals_overseg = error_proposals["proposals_overseg"]
            for proposal in proposals_overseg:
                candidates, i, j = proposal
                j -= 1
                mu = img_stack_meta_info["mus"][i][j:j+1]
                cov = img_stack_meta_info["covs"][i][j:j+1]
                mu_old = img_stack_meta_info["mus_old"][i][j:j+1]
                cov_old = img_stack_meta_info["covs_old"][i][j:j+1]
                prob = img_stack_meta_info["probs"][i][j:j+1]
                z_pot_overseg.append({
                    "candidates": candidates,
                    "splits": {
                        "z": Gaussians(
                            mu=mu, covariances=cov
                        ),
                        "z_old": Gaussians(
                            mu=mu_old, covariances=cov_old
                        ),
                        "lambda_c_j": 1 - prob
                    }
                })

            proposals_underseg = error_proposals["proposals_underseg"]
            for proposal in proposals_underseg:
                j, i, candidates = proposal
                mus = img_stack_meta_info["mus"][i]
                covs = img_stack_meta_info["covs"][i]
                mus_old = img_stack_meta_info["mus_old"][i]
                covs_old = img_stack_meta_info["covs_old"][i]
                probs = img_stack_meta_info["probs"][i]
                inds = [c - 1 for c in candidates]
                z_pot_underseg.append({
                    "z_id": j,
                    "splits": {
                        "z": Gaussians(
                            mu=mus[inds], covariances=covs[inds]
                        ),
                        "z_old": Gaussians(
                            mu=mus_old[inds], covariances=covs_old[inds]
                        ),
                        "lambda_c_j": 1 - probs[inds]
                    }
                })

        if len(z_pot_overseg) > 0 or len(z_pot_underseg) > 0:
            print("    ",
                  "Oversegmentations", len(z_pot_overseg),
                  "Undersegmentations", len(z_pot_underseg))

        ret["frame"] = idx
        ret["z"] = z
        ret["P_D"] = P_D
        ret["z_old"] = z_old
        ret["z_id"] = z_id
        ret["z_area"] = z_area
        ret["z_is_at_border"] = z_is_at_border
        ret["lambda_c_j"] = lambda_c_j
        ret["z_pot_overseg"] = z_pot_overseg
        ret["z_pot_underseg"] = z_pot_underseg

        if not only_gaussians:
            ret["img"] = tiff.imread(self.img_files[idx])
            ret["instances"] = tiff.imread(self.mask_files[idx])

        return ret


if __name__ == "__main__":
    data = CellTrackingChallengeSequence(
        path=r"C:\Users\kaiser\Desktop\data\CTC\Inference\normal_bayesian_no_postprocessing\train",
        dataset_name="BF-C2DL-HSC",
        sequence_name="02"
    )
