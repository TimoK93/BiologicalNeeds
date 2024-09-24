import os
from os.path import exists
import numpy as np
import argparse
import tifffile
from multiprocessing import Pool, cpu_count


def read_tracking_file(path):
    """
    Reads a text file representing an acyclic graph for the whole video.
    Every line corresponds to a single track that is encoded by four numbers
    separated by a space:
        L B E P where
        L - a unique label of the track (label of markers, 16-bit positive value)
        B - a zero-based temporal index of the frame in which the track begins
        E - a zero-based temporal index of the frame in which the track ends
        P - label of the parent track (0 is used when no parent is defined)

    # Checked against test datasets -> OK

    Args:
        path: Path to the text file.

    Returns:
        A numpy array of shape (N, 4) where N is the number of tracks.
        Each row represents a track and contains four numbers: L B E P
    """
    if not exists(path):
        return None
    with open(path, "r") as f:
        lines = f.readlines()
    lines = [x.strip().split(" ") for x in lines]
    lines = [[int(y) for y in x] for x in lines]
    return np.array(lines)


def relabel_file(file, res_path, out_path, id_mapping):
    img = tifffile.imread(os.path.join(res_path, file))
    img2 = np.copy(img)
    for l in np.unique(img)[1:]:
        if l not in id_mapping:
            print(f"Label {l} not found in mapping")
            print(file, res_path, out_path, id_mapping)
            print(np.unique(img))
            raise ValueError
        mapping = id_mapping[l]
        if l == mapping:
            continue
        img2[img == l] = id_mapping[l]
    tifffile.imwrite(os.path.join(out_path, file), img2)


def restore_new_result(
        tracks,
        files,
        res_path,
        out_path,
        id_mapping,
        multiprocessing=True,
):
    # Save new tracks
    np.savetxt(os.path.join(out_path, "res_track.txt"), tracks, delimiter=" ", fmt="%d")

    # Save new images
    args = [(file, res_path, out_path, id_mapping) for file in files]
    if multiprocessing:
        with Pool(cpu_count()) as pool:
            pool.starmap(relabel_file, args)
    else:
        for arg in args:
            relabel_file(*arg)


def postprocess(
        tracks,
        res_path,
        out_path,
        multiprocessing=True,
):
    if tracks.size == 0:
        tracks = np.zeros((0,8))
    # Merge nearest neighbors in close scenarios
    files = sorted([x for x in os.listdir(res_path) if x.endswith(".tif")])
    # Copy all files to the output directory
    for file in files:
        img = tifffile.imread(os.path.join(res_path, file))
        tifffile.imwrite(os.path.join(out_path, file), img)
    # Find tracks to merge
    unique, counts = np.unique(
        tracks[:, 3][tracks[:, 3] > 0], return_counts=True)
    occluded_tracks = unique[counts == 1]
    to_combine_tracks = []
    for track in occluded_tracks:
        id2 = int(tracks[tracks[:, 3] == track, 0].squeeze())
        end1 = int(tracks[tracks[:, 0] == track, 2].squeeze())
        start2 = int(tracks[tracks[:, 3] == track, 1].squeeze())
        to_combine_tracks.append(
            (track, id2, end1, start2)
        )

    # Add masks for tracks to merge
    maximal_distance = 10
    for track, id2, end1, start2 in to_combine_tracks:
        # Read masks of the two tracks
        mask1 = tifffile.imread(os.path.join(res_path, files[end1]))
        mask2 = tifffile.imread(os.path.join(res_path, files[start2]))
        h, w = mask1.shape
        # Extract regions of the two tracks
        binary_mask1 = mask1 == track
        binary_mask2 = mask2 == id2
        mask1 = np.argwhere(binary_mask1)
        mask2 = np.argwhere(binary_mask2)
        if mask1.size == 0 or mask2.size == 0:
            print("Empty mask", track, id2, end1, start2)
            continue
        x1, y1 = np.mean(mask1[:, 1]), np.mean(mask1[:, 0])
        x2, y2 = np.mean(mask2[:, 1]), np.mean(mask2[:, 0])
        # Compute distance between the two tracks
        dx, dy, dt = x2 - x1, y2 - y1, start2 - end1
        if dt > maximal_distance:
            continue
        for i in range(1, dt):
            if i < dt - 1:
                delta_mask = np.copy(mask1)
                offset_x = int(dx * i / dt)
                offset_y = int(dy * i / dt)
                delta_mask[:, 0] += offset_y
                delta_mask[:, 1] += offset_x
            else:
                delta_mask = np.copy(mask2)
                offset_x = int(dx * (dt - i) / dt)
                offset_y = int(dy * (dt - i) / dt)
                delta_mask[:, 0] -= offset_y
                delta_mask[:, 1] -= offset_x
            delta_mask = delta_mask[
                (delta_mask[:, 0] >= 0) & (delta_mask[:, 0] < h) &
                (delta_mask[:, 1] >= 0) & (delta_mask[:, 1] < w)
                ]
            img = tifffile.imread(os.path.join(out_path, files[end1 + i]))
            _img = np.zeros_like(img)
            _img[delta_mask[:, 0], delta_mask[:, 1]] = 1
            _img_filtered = (_img == 1) & (img == 0)
            remaining_ratio = np.sum(_img_filtered) / np.sum(_img)
            if not np.any(_img) or remaining_ratio < 0.85:
                break
            print(f"    Add mask at frame {end1 + i} with id {track}")
            img[_img_filtered] = track
            tifffile.imwrite(os.path.join(out_path, files[end1 + i]), img)
            tracks[tracks[:, 0] == track, 2] = end1 + i

    # Find tracks to merge after new masks have been added
    unique, counts = np.unique(
        tracks[:, 3][tracks[:, 3] > 0], return_counts=True)
    occluded_tracks = unique[counts == 1]
    to_combine_tracks = []
    for track in occluded_tracks:
        id2 = int(tracks[tracks[:, 3] == track, 0].squeeze())
        end1 = int(tracks[tracks[:, 0] == track, 2].squeeze())
        start2 = int(tracks[tracks[:, 3] == track, 1].squeeze())
        if start2 - end1 == 1:
            to_combine_tracks.append(
                (track, id2, end1, start2)
            )

    # Map old to new labels
    new_tracks = np.copy(tracks)
    id_mapping = {k: k for k in np.unique(new_tracks[:, 0])}
    id_mapping[0] = 0
    for track, id2, end1, start2 in to_combine_tracks:
        end2 = int(new_tracks[new_tracks[:, 0] == id_mapping[id2], 2].squeeze())
        new_tracks = new_tracks[new_tracks[:, 0] != id_mapping[id2]]
        new_tracks[new_tracks[:, 0] == track, 2] = end2
        new_tracks[new_tracks[:, 3] == id_mapping[id2], 3] = id_mapping[track]
        id_mapping[id2] = id_mapping[track]
        for k, v in id_mapping.items():
            if v == id2:
                id_mapping[k] = id_mapping[track]

    # Merge tracks for res file
    new_tracks = np.copy(tracks)
    remaining_ids = np.unique([int(k) for k in id_mapping.values()])
    keys = [int(k) for k in id_mapping.keys()]
    for i in remaining_ids:
        if i == 0:
            continue
        begin, end = 100000000, -1
        for j in np.unique(keys):
            if j == 0:
                continue
            j = int(j)
            if id_mapping[j] == i:
                _begin, _end = tracks[tracks[:, 0] == j, 1:3][0]
                begin = min(begin, _begin)
                end = max(end, _end)
        new_tracks[tracks[:, 0] == i, 1:3] = begin, end
    remains = np.isin(new_tracks[:, 0], remaining_ids)
    new_tracks = new_tracks[remains]

    # Fill missing track ids
    _id_mapping = {k: v for k, v in id_mapping.items()}
    order = np.argsort(new_tracks[:, 1])
    remaining_labels = new_tracks[order, 0]
    new_labels = np.arange(1, len(remaining_labels)+1)
    for old, new in zip(remaining_labels, new_labels):
        for k, v in id_mapping.items():
            if id_mapping[k] == old:
                _id_mapping[k] = new
    id_mapping = _id_mapping

    new_tracks[:, 0] = [int(id_mapping[k]) for k in new_tracks[:, 0]]
    new_tracks[:, 3] = [int(id_mapping[k]) for k in new_tracks[:, 3]]
    # Save new result
    restore_new_result(
        new_tracks, files, out_path, out_path, id_mapping, multiprocessing
    )

    return


def postprocess_all(
        data_root,
        dest_root,
        subset=None,
        dataset=None,
        sequence=None,
):
    for _subset in ["train", "challenge"]:
        if subset != _subset and subset is not None:
            continue
        datasets = os.listdir(os.path.join(data_root, _subset))
        for _dataset in datasets:
            if dataset != _dataset and dataset is not None:
                continue
            # if _dataset == "PhC-C2DL-PSC":
            #    continue
            for _sequence in ["01", "02"]:
                if sequence != _sequence and sequence is not None:
                    continue
                if not os.path.exists(os.path.join(
                        data_root, _subset, _dataset, _sequence+"_RES")
                ):
                    continue
                print(f"Interpolate {_subset} {_dataset} {_sequence}")
                postprocess_sequence(
                    data_root=os.path.join(data_root, _subset),
                    dest_root=os.path.join(dest_root, _subset),
                    dataset_name=_dataset,
                    sequence_name=_sequence,
                )


def postprocess_sequence(
        data_root,
        dest_root,
        dataset_name,
        sequence_name,
):
    in_dir = os.path.join(data_root, dataset_name, sequence_name + "_RES")
    res_dir = os.path.join(dest_root, dataset_name, sequence_name + "_RES")
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    tracks = read_tracking_file(os.path.join(in_dir, "res_track.txt"))
    if tracks is None:
        print("WARNING: No res file in ", in_dir)
        return
    postprocess(
        tracks,
        in_dir,
        res_dir,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Interpolation of a tracking result'
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
            dest_root=args.destination_root,
            dataset_name=args.dataset,
            sequence_name=args.sequence,
        )
    else:
        postprocess_all(
            data_root=args.data_root,
            dest_root=args.destination_root,
            subset=args.subset,
            dataset=args.dataset,
            sequence=args.sequence,
        )
