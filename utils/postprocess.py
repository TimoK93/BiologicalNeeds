import os
import numpy as np
import argparse
import tifffile
import re
import pandas as pd
from pathlib import Path
import shutil

from embedtrack.infer.infer_ctc_data import fill_empty_frames


""" 
###########################################
Third party code from EmbedTrack Repository 
###########################################
"""


def get_indices_pandas(data, background_id=0):
    """
    Extracts for each mask id its positions within the array.
    Args:
        data: a np. array with masks, where all pixels belonging to the
            same masks have the same integer value
        background_id: integer value of the background

    Returns: data frame: indices are the mask id , values the positions of the mask pixels

    """
    if data.size < 1e9:  # aim for speed at cost of high memory consumption
        masked_data = data != background_id
        flat_data = data[masked_data]  # d: data , mask attribute
        dummy_index = np.where(masked_data.ravel())[0]
        df = pd.DataFrame.from_dict({"mask_id": flat_data, "flat_index": dummy_index})
        df = df.groupby("mask_id", group_keys=True).apply(
            lambda x: np.unravel_index(x.flat_index, data.shape)
        )
    else:  # aim for lower memory consumption at cost of speed
        flat_data = data[(data != background_id)]  # d: data , mask attribute
        dummy_index = np.where((data != background_id).ravel())[0]
        mask_indices = np.unique(flat_data)
        df = {"mask_id": [], "index": []}
        data_shape = data.shape
        for mask_id in mask_indices:
            df["index"].append(
                np.unravel_index(dummy_index[flat_data == mask_id], data_shape)
            )
            df["mask_id"].append(mask_id)
        df = pd.DataFrame.from_dict(df)
        df = df.set_index("mask_id")
        df = df["index"].apply(lambda x: x)  # convert to same format as for other case
    return df


def get_img_files(img_dir, starts_with=""):
    """
    Extracts a set of tiff files from a folder.
    Args:
        img_dir: path to the image folder
        starts_with: optional string the image name needs to start with

    Returns:

    """
    img_file_pattern = re.compile(
        r"(\D*)(\d+)(\D*)\.(" + "|".join(("tif", "tiff")) + ")"
    )
    files = {
        int(img_file_pattern.match(file).groups()[1]): (img_dir / file).as_posix()
        for file in sorted(os.listdir(img_dir))
        if file.endswith(("tif", "tiff")) and file.startswith(starts_with)
    }
    return files


def find_tracklets(data_path, lineage_name="res_track.txt"):
    """
    Checks tracking results for inconsistencies between lineage file and tracking masks.
    Args:
        data_path: string or posix path to tracking directory
    """
    data_path = Path(data_path)
    lineage_data = pd.read_csv(
        (data_path / lineage_name).as_posix(),
        delimiter=" ",
        names=["m_id", "t_s", "t_e", "pred"],
        header=None,
        index_col=0,
    )
    img_files = get_img_files(Path(data_path))

    tracks = {}
    for time_point in sorted(img_files.keys()):
        tracking_masks = get_indices_pandas(tifffile.imread(img_files[time_point]))
        for mask_id in tracking_masks.index:
            if mask_id not in tracks:
                tracks[mask_id] = []
            tracks[mask_id].append(time_point)

    assert not set(tracks.keys()).symmetric_difference(
        set(lineage_data.index)
    ), f"masks in segm images: {set(tracks.keys())} and tra file {set(lineage_data.index)} not consistent, " \
       f"{np.setdiff1d(list(tracks.keys()), list(lineage_data.index))} missing in lineage file, " \
         f"{np.setdiff1d(list(lineage_data.index), list(tracks.keys()))} missing in segm images"

    tracklets = {}
    for track_id, time_points in tracks.items():
        time_points = np.array(sorted(time_points))
        t_min, t_max = min(time_points), max(time_points)
        missing = t_max - t_min + 1 - len(time_points)
        if missing:
            d_t = time_points[1:] - time_points[:-1] - 1
            t_start_tracklets = [time_points[0], *time_points[1:][d_t > 0]]
            t_end_tracklets = [*time_points[:-1][d_t > 0], time_points[-1]]
            tracklets[track_id] = list(zip(t_start_tracklets, t_end_tracklets))
    return tracklets


def edit_tracks_with_missing_masks(tracking_dir):
    fragmented_tracks = find_tracklets(tracking_dir, "res_track.txt")
    if len(fragmented_tracks) == 0:
        return

    lineage_data = pd.read_csv(
        os.path.join(tracking_dir, "res_track.txt"),
        delimiter=" ",
        names=["m_id", "t_s", "t_e", "pred"],
        header=None,
        index_col=None,
    )
    # track_id : [t_start(gap), gap_length]
    mask_files = {
        int(re.findall("\d+", file_name)[0]): os.path.join(tracking_dir, file_name)
        for file_name in os.listdir(tracking_dir)
        if file_name.endswith(".tif")
    }

    max_track_id = lineage_data["m_id"].max()
    for track_id, t_tracklets in fragmented_tracks.items():

        tracklet_parent = 0
        for i_tracklet, (t_start, t_end) in enumerate(
            t_tracklets
        ):  # keep first tracklet unchanged as we can keep the original track id for it
            if i_tracklet == 0:
                lineage_data.loc[lineage_data["m_id"] == track_id, "t_e"] = t_end
                tracklet_parent = track_id
                continue

            max_track_id += 1
            for t in range(t_start, t_end + 1):
                mask_img = tifffile.imread(mask_files[t])
                mask_img[mask_img == track_id] = max_track_id
                tifffile.imwrite(mask_files[t], mask_img)
            lineage_data = pd.concat([lineage_data, pd.DataFrame([{
                    "m_id": max_track_id,
                    "t_s": t_start,
                    "t_e": t_end,
                    "pred": tracklet_parent,
                }])], ignore_index=True)
            # lineage_data = lineage_data.append(
            #     {
            #         "m_id": max_track_id,
            #         "t_s": t_start,
            #         "t_e": t_end,
            #         "pred": tracklet_parent,
            #     },
            #     ignore_index=True,
            # )
            tracklet_parent = max_track_id

    lineage_data.to_csv(
        os.path.join(tracking_dir, "res_track.txt"), sep=" ", index=False, header=False
    )


def remove_missing_tracks_from_lineage(tracking_dir):
    lineage_data = pd.read_csv(
        os.path.join(tracking_dir, "res_track.txt"),
        delimiter=" ",
        names=["m_id", "t_s", "t_e", "pred"],
        header=None,
        index_col=None,
    )
    tracks = {}
    for time_idx, mask_file in get_img_files(Path(tracking_dir)).items():
        mask_ids = np.unique(tifffile.imread(mask_file))
        mask_ids = mask_ids[mask_ids != 0]
        for mask_idx in mask_ids:
            if mask_idx not in tracks.keys():
                tracks[mask_idx] = []
            tracks[mask_idx].append(time_idx)
    missing_masks = set(lineage_data["m_id"].values).symmetric_difference(
        set(tracks.keys())
    )
    for mask_idx in missing_masks:
        lineage_data.loc[lineage_data["pred"] == mask_idx, "pred"] = 0
        lineage_data = lineage_data[lineage_data["m_id"] != mask_idx]
    for mask_idx, time_steps in tracks.items():
        lineage_data.loc[lineage_data["m_id"] == mask_idx, "t_s"] = min(time_steps)
        lineage_data.loc[lineage_data["m_id"] == mask_idx, "t_e"] = max(time_steps)
    lineage_data.to_csv(
        os.path.join(tracking_dir, "res_track.txt"), sep=" ", index=False, header=False
    )


def foi_correction(tracking_dir, cell_type):
    """
    Adapt data sets using the field of interest definition of the CTC.
    Args:
        tracking_dir: string
            path where the segmented (and tracked) data is stored
        cell_type: string
            name of the data set

    Returns:

    """
    border_with = 0
    if cell_type in [
        "DIC-C2DH-HeLa",
        "Fluo-C2DL-Huh7",
        "Fluo-C2DL-MSC",
        "Fluo-C3DH-H157",
        "Fluo-N2DH-GOWT1",
        "Fluo-N3DH-CE",
        "Fluo-N3DH-CHO",
        "PhC-C2DH-U373",
    ]:
        border_with = 50
    if cell_type in [
        "BF-C2DL-HSC",
        "BF-C2DL-MuSC",
        "Fluo-C3DL-MDA231",
        "Fluo-N2DL-HeLa",
        "PhC-C2DL-PSC",
    ]:
        border_with = 25

    for segm_file in os.listdir(tracking_dir):
        if not segm_file.endswith(".tif"):
            continue
        segm_mask = tifffile.imread(os.path.join(tracking_dir, segm_file))
        if len(segm_mask.shape) == 2:
            segm_mask_foi = segm_mask[
                border_with : segm_mask.shape[0] - border_with,
                border_with : segm_mask.shape[1] - border_with,
            ]
        else:
            segm_mask_foi = segm_mask[
                :,
                border_with : segm_mask.shape[1] - border_with,
                border_with : segm_mask.shape[2] - border_with,
            ]
        idx_segm_mask = np.unique(segm_mask)
        idx_segm_mask = idx_segm_mask[idx_segm_mask > 0]
        idx_segm_mask_foi = np.unique(segm_mask_foi)
        idx_segm_mask_foi = idx_segm_mask_foi[idx_segm_mask_foi > 0]
        missing_masks = idx_segm_mask[~np.isin(idx_segm_mask, idx_segm_mask_foi)]
        for mask_id in missing_masks:
            segm_mask[segm_mask == mask_id] = 0
        tifffile.imwrite(
            os.path.join(tracking_dir, segm_file), segm_mask.astype(np.uint16)
        )
    # remove all now missing tracks from the lineage
    remove_missing_tracks_from_lineage(tracking_dir)
    # reload all mask files -> get fragmented tracks and rename them
    edit_tracks_with_missing_masks(tracking_dir)


def fill_empty_frames(mask_dir):
    """
    Adds for each empty tracking frame the tracking result of the temporally closest frame.
    Otherwise CTC measure can yield an error.
    Args:
        tracks: a dict containing the tracking results
        time_steps: a list of time steps

    Returns: the modified tracks

    """
    time_steps = [
        (time_idx, file) for time_idx, file in get_img_files(Path(mask_dir)).items()
    ]
    time_steps.sort(key=lambda x: x[0])
    filled_time_steps = []
    empty_time_steps = []
    for time, file in time_steps:
        segm_mask = tifffile.imread(file)
        mask_ids = np.unique(segm_mask)
        mask_ids = mask_ids[mask_ids != 0]
        if len(mask_ids) > 0:
            filled_time_steps.append((time, file))
        else:
            empty_time_steps.append((time, file))

    filled_t, filled_files = list(zip(*filled_time_steps))

    lineage = pd.read_csv(
        os.path.join(mask_dir, "res_track.txt"),
        delimiter=" ",
        header=None,
        names=["cell_id", "t_start", "t_end", "predecessor"],
    )
    lineage = lineage.set_index("cell_id")
    for empty_t, empty_file in empty_time_steps:
        nearest_filled_frame = filled_files[
            np.argmin(abs(np.array(filled_t) - empty_t))
        ]
        os.remove(empty_file)
        shutil.copyfile(nearest_filled_frame, empty_file)
        new_masks = tifffile.imread(empty_file)
        mask_ids = np.unique(new_masks)
        mask_ids = mask_ids[mask_ids != 0]
        for mask_idx in mask_ids:
            lineage.loc[mask_idx, ("t_start")] = min(
                empty_t, lineage.loc[mask_idx]["t_start"]
            )
            lineage.loc[mask_idx, ("t_end")] = max(
                empty_t, lineage.loc[mask_idx]["t_end"]
            )

    lineage = lineage.reset_index().sort_values("cell_id")
    lineage.to_csv(
        os.path.join(mask_dir, "res_track.txt"), sep=" ", index=False, header=False
    )


"""
###########################################
"""


def postprocess_all(
        data_root,
        subset=None,
        dataset=None,
        sequence=None,
):
    for _subset in ["train", "challenge"]:
        if subset != _subset and subset is not None:
            continue
        if not os.path.exists(os.path.join(data_root, _subset)):
            continue
        datasets = os.listdir(os.path.join(data_root, _subset))
        for _dataset in datasets:
            if dataset != _dataset and dataset is not None:
                continue
            for _sequence in ["01", "02"]:
                if sequence != _sequence and sequence is not None:
                    continue
                if not os.path.exists(os.path.join(
                        data_root, _subset, _dataset, _sequence+"_RES")
                ):
                    continue
                print(f"Process {_subset} {_dataset} {_sequence}")
                postprocess_sequence(
                    data_root=os.path.join(data_root, _subset),
                    dataset_name=_dataset,
                    sequence_name=_sequence,
                )


def postprocess_sequence(
        data_root,
        dataset_name,
        sequence_name,
):
    res_dir = os.path.join(data_root, dataset_name, sequence_name + "_RES")
    try:
        foi_correction(res_dir, dataset_name)
        fill_empty_frames(res_dir)
    except Exception as e:
       print("WARNING: There was an error in utils sequence", res_dir)
       print(e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Postprocessing according to CTC conformity. Cloned from'
                    'original EmbedTrack solution.'
    )
    parser.add_argument(
        '--data-root', required=True,
        help="Path to the tracking data root"
    )
    parser.add_argument(
        '--subset', default=None,
        help="train or challenge, if None both are evaluated"
    )
    parser.add_argument(
        '--dataset', default=None,
        help="CTC-Challenge, if None all are evaluated"
    )
    parser.add_argument(
        '--sequence', default=None,
        help="01 or 02, if None both are evaluated"
    )
    parser.add_argument(
        '--single-sequence', action="store_true",
        help="Only infer a single sequence"
    )

    args = parser.parse_args()

    if args.single_sequence:
        postprocess_sequence(
            data_root=args.data_root,
            dataset_name=args.dataset,
            sequence_name=args.sequence,
        )
    else:
        postprocess_all(
            data_root=args.data_root,
            subset=args.subset,
            dataset=args.dataset,
            sequence=args.sequence,
        )
