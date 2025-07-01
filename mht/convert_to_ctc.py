import numpy as np
import os
import tifffile as tiff
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from os.path import join, exists
import pickle

MULTIPROCESSING = True


def restore_hypothesis(data, hypothesis_id=None, frame=None):
    """

    Args:
        data: Result data of the tracker
        hypothesis_id: Define a specific hypothesis to reconstruct. If None, the
            most likely hypothesis is chosen.
        frame: Define a specific frame to end reconstruction. If None, the last
            frame is chosen.

    Returns:
        A list of all single frame states that belong to the hypothesis.

    """
    frame = frame if frame is not None else len(data["hypotheses"]) - 1
    if hypothesis_id is None:
        hypotheses = data["hypotheses"][frame]
        log_likelihood = [h["l"] for h in hypotheses]
        hypothesis_id = hypotheses[np.argmax(log_likelihood)]["id"]
    track_hypothesis = list()
    parent_hypothesis = hypothesis_id
    for frame in reversed(data["hypotheses"]):
        for hypothesis in frame:
            if hypothesis["id"] == parent_hypothesis:
                track_hypothesis.append(hypothesis)
                parent_hypothesis = hypothesis["parent"]
                break
    track_hypothesis.reverse()
    return track_hypothesis


def hypothesis_to_ndarray(hypothesis):
    '''
    n x (frame, mu_x, mu_y, age, r, label, parent_label, associated_id)
    :param hypothesis:
    :return:
    '''

    a = list()
    for f, frame in enumerate(hypothesis):
        a.append(np.concatenate([
            np.ones_like(frame["r"]) * f,
            frame["mu"],
            frame["age"],
            frame["r"],
            frame["labels"],
            frame["parent_label"],
            frame["associated_id"]
        ], axis=1))
    a = np.concatenate(a, axis=0)
    a[:, 6] = np.maximum(a[:, 6], 0)
    return a


def create_mask(arg):
    in_file, in_dir, res_dir, h = arg
    frame = int(in_file[len("mask"):-(len(".tif"))])
    in_path = join(in_dir, in_file)
    meta_file_path = join(in_dir, in_file.replace("mask", "t") + ".pkl")
    meta_data = None
    if exists(meta_file_path):
        with open(meta_file_path, "rb") as file:
            meta_data = pickle.load(file)
    res_path = join(res_dir, in_file)
    old = tiff.imread(in_path)
    new = np.zeros(old.shape, dtype=np.uint16)
    height, width = old.shape
    objects = np.where((h[:, 0] == frame) & (h[:, 7] != 0))[0]
    unique_assigned_ids, counts = np.unique(h[objects, 7], return_counts=True)
    undersegs = unique_assigned_ids[counts > 1]
    unique_object_ids, counts = np.unique(h[objects, 5], return_counts=True)
    oversegs = unique_object_ids[counts > 1]
    false_negative_ids = np.where((h[:, 0] == frame) & (h[:, 7] >= 1000000))[0]
    # Oversegmented objects
    # Undersegmented objects
    for m in undersegs:
        # Try to resolve multi assignments with precomputed masks
        idx = np.where((h[:, 7] == m) & (h[:, 0] == frame))[0]
        is_processed = False
        if meta_data is not None:
            if "img_stack_meta_info" in meta_data["object_states"]:
                masks = np.load(meta_data['object_states']["img_stack"])
                masks = masks["arr_0"]
                masks_meta = meta_data['object_states']["img_stack_meta_info"]
                proposals = meta_data["object_states"]["error_proposals"]
                proposals = proposals["proposals_underseg"]
                # Check if matching proposal is existing
                matches = [p for p in proposals if p[0] == m and len(p[2]) == len(idx)]
                if len(matches) > 0:
                    j, i, candidates = matches[0]
                    inds = [c - 1 for c in candidates]
                    mus_p = masks_meta["mus"][i][inds]
                    mus_m = h[idx, 1:3]
                    args_p = np.argsort(np.sum(mus_p, axis=1))
                    args_m = np.argsort(np.sum(mus_m, axis=1))
                    inds = np.asarray(inds)[args_p]
                    idx = idx[args_m]
                    mus_p = mus_p[args_p]
                    mus_m = mus_m[args_m]
                    is_matched = np.allclose(mus_p, mus_m, atol=0.0001)
                    if is_matched:
                        is_processed = True
                        for o, n in zip(inds, idx):
                            roi = masks[i] == o + 1
                            n_id = int(h[n, 5])
                            new[roi] = n_id
        if is_processed:
            continue
        # # Solve heuristically with nearest neighbour matching
        # x1, y1, age1, r1, label1, parent_label1, associated_id1 = \
        #     h[idx[0], 1:]
        # x2, y2, age2, r2, label2, parent_label2, associated_id2 = \
        #     h[idx[1], 1:]
        # x1, x2, y1, y2 = x1 * width, x2 * width, y1 * height, y2 * height
        # area = np.argwhere(old == m)
        # dist1 = np.linalg.norm(area - np.asarray([y1, x1]), axis=1)
        # dist2 = np.linalg.norm(area - np.asarray([y2, x2]), axis=1)
        # belongs_to_1 = dist1 < dist2
        # assert np.count_nonzero(belongs_to_1) < len(area)
        # assert np.count_nonzero(~belongs_to_1) < len(area)
        # new[area[belongs_to_1, 0], area[belongs_to_1, 1]] = int(label1)
        # new[area[~belongs_to_1, 0], area[~belongs_to_1, 1]] = int(label2)
    # Oversegmented objects
    for i in oversegs:
        i = int(i)
        x, y, age, r, label, parent_label, associated_id = h[i, 1:]
        new[old == associated_id] = int(label)
    # Normal objects
    for i in objects:
        x, y, age, r, label, parent_label, associated_id = h[i, 1:]
        if associated_id in undersegs:
            continue
        if associated_id in false_negative_ids:
            continue
        if i in oversegs:
            continue
        new[old == associated_id] = int(label)
    # False negative objects
    for i in false_negative_ids:
        print("Add false negative")
        if meta_data is None:
            continue
        x, y, age, r, label, parent_label, associated_id = h[i, 1:]
        slice = (associated_id // 1000000) - 1
        associated_id = associated_id % 1000000
        new[masks[slice] == associated_id] = int(label)
    tiff.imwrite(res_path, new)


def find_objects(arg):
    distance_threshold = 50
    in_file, in_dir, h, parents, children, positions = arg
    frame = int(in_file[len("mask"):-(len(".tif"))])
    in_path = join(in_dir, in_file)
    meta_file_path = join(in_dir, in_file.replace("mask", "t") + ".pkl")
    if exists(meta_file_path):
        with open(meta_file_path, "rb") as file:
            meta_data = pickle.load(file)
        objects_states = meta_data["object_states"]
        if "img_stack_meta_info" not in objects_states:
            proposals = []
            masks_meta = None
        else:
            masks_meta = objects_states["img_stack_meta_info"]
            proposals = objects_states["error_proposals"]
            #proposals = proposals["proposals_false_neg"]
            proposals = []
        if len(proposals) > 0:
            proposal_ids = [p[0] for p in proposals]
            proposal_ids = np.asarray(proposal_ids)
            proposal_frames = [p[1] for p in proposals]
            proposal_frames = np.asarray(proposal_frames)
            proposal_mus = [
                masks_meta["mus"][f][i-1] for i, f in zip(proposal_ids,
                                                         proposal_frames)]
            proposal_mus = np.stack(proposal_mus, axis=0)
        else:
            proposal_ids = np.zeros(0)
            proposal_frames = np.zeros(0)
            proposal_mus = np.zeros((0, 2))
    else:
        return None, None
    original_mask = tiff.imread(in_path)
    height, width = original_mask.shape[:2]
    # Extract potential mask to match
    mask_labels = np.unique(original_mask)
    mask_labels = mask_labels[mask_labels > 0]
    h = h[(h[:, 0] == frame) & (h[:, 7] != 0)]
    assigned_ids, counts = np.unique(h[:, 7], return_counts=True)
    unassigned_label_ids = np.setdiff1d(mask_labels, assigned_ids)
    if unassigned_label_ids.size > 0:
        unassigned_label_inds = [
            int(np.where(objects_states["id"] == l)[0]) for l in unassigned_label_ids]
        unassigned_label_inds = np.asarray(unassigned_label_inds)
        x = [objects_states["x"][i] for i in unassigned_label_inds]
        y = [objects_states["y"][i] for i in unassigned_label_inds]
        unassigned_label_mus = np.stack([[x, y] for x, y in zip(x, y)], axis=0)
    else:
        unassigned_label_inds = np.zeros(0)
        unassigned_label_mus = np.zeros((0, 2))
    overseg_label_ids, counts = np.unique(h[:, 5], return_counts=True)
    overseg_label_ids = overseg_label_ids[counts > 1]
    overseg_label_ids = h[np.isin(h[:, 5], overseg_label_ids), 7]
    if overseg_label_ids.size > 0:
        overseg_label_inds = [
            int(np.where(objects_states["id"] == l)[0]) for l in overseg_label_ids]
        overseg_label_inds = np.asarray(overseg_label_inds)
        x = [objects_states["x"][i] for i in overseg_label_inds]
        y = [objects_states["y"][i] for i in overseg_label_inds]
        overseg_label_mus = np.stack([[x, y] for x, y in zip(x, y)], axis=0)
    else:
        overseg_label_inds = np.zeros(0)
        overseg_label_mus = np.zeros((0, 2))
    # Try to find potential unassigned objects
    new_objects = list()
    to_remove_objects = list()
    for p, c, pos, in zip(parents, children, positions):
        # Check in unassigned masks
        diff = unassigned_label_mus - pos[None]
        diff[:, 0] *= width
        diff[:, 1] *= height
        dists = np.linalg.norm(diff, axis=1)
        if np.any(dists < distance_threshold):
            # Create new object
            i = np.argmin(dists)
            assigned_id = unassigned_label_ids[i]
            new_object = [frame, pos[0], pos[1], 1, 1, c, p, assigned_id]
            new_objects.append(np.asarray(new_object)[None, :])
            # Remove the assigned label from candidates lists
            inds = unassigned_label_ids != assigned_id
            unassigned_label_mus = unassigned_label_mus[inds]
            unassigned_label_inds = unassigned_label_inds[inds]
            unassigned_label_ids = unassigned_label_ids[inds]
            continue
        # Check in overseg masks
        diff = overseg_label_mus - pos[None]
        diff[:, 0] *= width
        diff[:, 1] *= height
        dists = np.linalg.norm(diff, axis=1)
        if np.any(dists < distance_threshold):
            # Create new object
            i = np.argmin(dists)
            assigned_id = overseg_label_ids[i]
            new_object = [frame, pos[0], pos[1], 1, 1, c, p, assigned_id]
            new_objects.append(np.asarray(new_object)[None, :])
            # Remove the assigned label from candidates lists
            inds = overseg_label_ids != assigned_id
            overseg_label_mus = overseg_label_mus[inds]
            overseg_label_inds = overseg_label_inds[inds]
            overseg_label_ids = overseg_label_ids[inds]
            # Add old assignment to remove list
            to_remove_objects.append(
                h[(h[:, 0] == frame) & (h[:, 7] == assigned_id)])
            continue
        # Check in proposals
        diff = proposal_mus - pos[None]
        diff[:, 0] *= width
        diff[:, 1] *= height
        dists = np.linalg.norm(diff, axis=1)
        if np.any(dists < distance_threshold):
            # Create new object
            i = np.argmin(dists)
            assigned_id = proposal_ids[i]
            f = proposal_frames[i]
            mapped_id = 1000000 * (f + 1) + assigned_id
            new_object = [frame, pos[0], pos[1], 1, 1, c, p, mapped_id]
            new_objects.append(np.asarray(new_object)[None, :])
            # Remove the assigned label from candidates lists
            inds = proposal_ids != assigned_id
            proposal_mus = proposal_mus[inds]
            proposal_ids = proposal_ids[inds]
            proposal_frames = proposal_frames[inds]
            continue

    if len(new_objects) == 0:
        new_objects = None
    elif len(new_objects) == 1:
        new_objects = new_objects[0]
    else:
        new_objects = np.concatenate(new_objects, axis=0)

    if len(to_remove_objects) == 0:
        to_remove_objects = None
    elif len(to_remove_objects) == 1:
        to_remove_objects = to_remove_objects[0]
    else:
        to_remove_objects = np.concatenate(to_remove_objects, axis=0)

    return new_objects, to_remove_objects


def convert_to_ctc(
        in_dir, res_dir,
        h,
        r_threshold=0.01,
        split_trajectory=True,
        min_mean_trajectory_r=0.5,
        min_trajectory_length=3,

):
    '''

    :param in_dir:
    :param res_dir:
    :param h: hypothesis as np.ndarray with
        n x (frame, mu_x, mu_y, age, r, label, parent_label, associated_id)
    :param r_threshold: minimum confidence to be considered
    :param split_trajectory: if a trajectory is not present in a frame, it will
        be split and the new trajectory will be a daughter cell assigned to a
        new id.
    :return:
    '''
    # List files
    in_files = [x for x in sorted(os.listdir(in_dir)) if x.endswith(".tif")]
    # Create result directory
    assert exists(in_dir)
    os.makedirs(res_dir, exist_ok=True)
    # Sort hypothesis by frame
    inds = np.argsort(h[:, 0])
    h = h[inds]
    max_frame = int(np.max(h[:, 0]))
    # Extract motions
    motions = dict()
    for i in np.unique(h[:, 5]).astype(int):
        trajectory = h[h[:, 5] == i]
        if len(trajectory) < 2:
            motions[i] = np.zeros(1)
            continue
        motions[i] = np.linalg.norm(
            trajectory[1:, 1:3] - trajectory[:-1, 1:3], axis=1)
    # Filter anomalous trajectories
    to_remove_ids = list()
    for i in np.unique(h[:, 5]).astype(int):
        trajectory = h[h[:, 5] == i]
        at_begin = np.any(trajectory[:, 0] == 0)
        at_end = np.any(trajectory[:, 0] == max_frame)
        # Check if trajectory is too short
        if len(trajectory) < min_trajectory_length and \
                not at_end and not at_begin:
            to_remove_ids.append(i)
            continue
        # Check if trajectory has too low mean probability (r)
        if np.mean(trajectory[:, 4]) < min_mean_trajectory_r:
            to_remove_ids.append(i)
            continue
        # Check if trajectory has no motion
        motion = motions[i]
        maximum = np.max(motion)
        if maximum == 0:
            to_remove_ids.append(i)
            continue
        # Remove trajectories that are 50% not visible
        _trajectory = np.copy(trajectory)
        for _ in range(len(_trajectory)):
            if _trajectory[-1, 7] <= 0:
                _trajectory = _trajectory[:-1]
            else:
                break
        unassociated_ids = np.sum(_trajectory[:, 7] <= 0).astype(int)
        if unassociated_ids / len(_trajectory) > 0.6:
            to_remove_ids.append(i)
            continue
    for i in to_remove_ids:
        h = h[h[:, 5] != i]
        h[h[:, 6] == i, 6] = 0
    h = h[h[:, 4] > r_threshold]  # Remove low confidence detections
    h = h[h[:, 7] > 0]  # Remove unassigned detections
    # Fill missing labels
    labels = np.unique(h[:, 5]).astype(int)
    max_label = int(np.max(labels)) if len(labels) > 0 else 0
    if max_label - 1 != labels.size:
        for l in reversed(np.setdiff1d(np.arange(1, max_label + 1), labels)):
            h[h[:, 5] > l, 5] -= 1
            h[h[:, 6] > l, 6] -= 1
    # Split trajectories that have gaps
    if split_trajectory:
        # Find trajectories with missing frames
        labels = np.unique(h[:, 5]).astype(int)
        next_label = int(np.max(labels) + 1) if len(h) > 0 else 1
        for l in labels:
            # Find frames between start and end where the trajectory is missing
            idx = np.where(h[:, 5] == l)[0]
            start, end = np.min(h[idx, 0]), np.max(h[idx, 0])
            missing_frames = np.setdiff1d(np.arange(start, end + 1), h[idx, 0])
            missing_frames = np.asarray(
                [f for f in missing_frames if f-1 not in missing_frames])
            if missing_frames.size > 0:
                current_l = l
                for f in missing_frames:
                    # Create new hypothesis
                    inds = (h[:, 0] > f) & (h[:, 5] == current_l)
                    daughters = h[:, 6] == current_l
                    h[inds, 5] = next_label
                    h[inds, 6] = current_l
                    h[daughters, 6] = next_label
                    current_l = next_label
                    next_label += 1
    # Fill missing labels
    labels = np.unique(h[:, 5]).astype(int)
    max_label = int(np.max(labels)) if len(labels) > 0 else 0
    if max_label - 1 != labels.size:
        for l in reversed(np.setdiff1d(np.arange(1, max_label + 1), labels)):
            h[h[:, 5] > l, 5] -= 1
            h[h[:, 6] > l, 6] -= 1
    # Sort ids by birth frame
    ids = np.unique(h[:, 5]).astype(int)
    births = np.asarray([np.min([h[h[:, 5] == id, 0]]) for id in ids])
    sorted_births = np.argsort(births)
    _h = np.copy(h)
    for i, idx in enumerate(sorted_births):
        id = ids[idx]
        _h[h[:, 5] == id, 5] = i + 1
        _h[h[:, 6] == id, 6] = i + 1
    h = _h
    h = h[np.argsort(h[:, 0])]
    # Try to find and "interpolate" missing objects
    parent_labels = np.unique(h[h[:, 6] > 0, 6])
    num_children = np.asarray(
        [len(np.unique(h[h[:, 6] == p, 5])) for p in parent_labels])
    parent = parent_labels[num_children == 1]
    child = [h[h[:, 6] == p, 5][0] for p in parent]
    missing_frames = list()
    positions = list()
    for p, c in zip(parent, child):
        end_parent = h[h[:, 5] == p][-1, 0]
        start_child = h[h[:, 5] == c][0, 0]
        pos_parent = h[h[:, 5] == p][-1, 1:3]
        pos_child = h[h[:, 5] == c][0, 1:3]
        time_gap = int(start_child - end_parent)
        x = np.linspace(pos_parent[0], pos_child[0], time_gap + 1)
        y = np.linspace(pos_parent[1], pos_child[1], time_gap + 1)
        positions.append(np.stack([x, y], axis=1)[1:-1])
        missing_frames.append(np.arange(end_parent + 1, start_child))
    args = [(in_file, in_dir, h, [], [], []) for in_file in in_files]
    for p, c, m, x in zip(parent, child, missing_frames, positions):
        for i, f in enumerate(m):
            args[int(f)][3].append(p)
            args[int(f)][4].append(c)
            args[int(f)][5].append(x[i])
    new_objects, to_remove_objects = list(), list()
    for arg in tqdm(args, "Interpolating missing objects"):
        if len(arg[3]) > 0:
            new, remove = find_objects(arg)
            if new is not None:
                new_objects.append(new)
            if remove is not None:
                to_remove_objects.append(remove)
    for o in to_remove_objects:
        h = h[~((h[:, 0] == o[0, 0]) & (h[:, 5] == o[0, 5]) & (h[:, 7] == o[0, 7]))]
    if len(new_objects) == 1:
        new_objects = new_objects[0]
    elif len(new_objects) > 1:
        new_objects = np.concatenate(new_objects, axis=0)
    else:
        new_objects = np.zeros((0, 8))
    if len(new_objects) > 0:
        h = np.concatenate([h, new_objects], axis=0)
        h = h[np.argsort(h[:, 5])]
        h = h[np.argsort(h[:, 0])]
    # Split trajectories that have gaps that potentially could be there now
    labels = np.unique(h[:, 5]).astype(int)
    next_label = int(np.max(labels) + 1) if len(h) > 0 else 1
    for l in labels:
        # Find frames between start and end where the trajectory is missing
        idx = np.where(h[:, 5] == l)[0]
        start, end = np.min(h[idx, 0]), np.max(h[idx, 0])
        missing_frames = np.setdiff1d(np.arange(start, end + 1), h[idx, 0])
        missing_frames = np.asarray(
            [f for f in missing_frames if f - 1 not in missing_frames])
        if missing_frames.size > 0:
            current_l = l
            for f in missing_frames:
                # Create new hypothesis
                inds = (h[:, 0] > f) & (h[:, 5] == current_l)
                daughters = h[:, 6] == current_l
                h[inds, 5] = next_label
                h[inds, 6] = current_l
                h[daughters, 6] = next_label
                current_l = next_label
                next_label += 1
    # Check if gaps are filled and correct the label assignment
    while True:
        parent_labels = np.unique(h[h[:, 6] > 0, 6])
        num_children = np.asarray(
            [len(np.unique(h[h[:, 6] == p, 5])) for p in parent_labels])
        parent = parent_labels[num_children == 1]
        child = [h[h[:, 6] == p, 5][0] for p in parent]
        closed_gaps = list()
        for p, c in zip(parent, child):
            end_parent = h[h[:, 5] == p][-1, 0]
            start_child = h[h[:, 5] == c][0, 0]
            if start_child - end_parent == 1:
                closed_gaps.append((p, c))
        if len(closed_gaps) == 0:
            break
        else:
            # Merge labels of closed gaps
            for p, c in reversed(closed_gaps):
                h[h[:, 5] == c, 5] = p
                h[h[:, 6] == c, 6] = p

    # Remap unique labels to replace missing labels
    labels = np.unique(h[:, 5]).astype(int)
    max_label = int(np.max(labels)) if len(labels) > 0 else 0
    if max_label - 1 != labels.size:
        for l in reversed(np.setdiff1d(np.arange(1, max_label + 1), labels)):
            h[h[:, 5] > l, 5] -= 1
            h[h[:, 6] > l, 6] -= 1

    # Create masks
    args = list()
    for in_file in in_files:
        args.append((in_file, in_dir, res_dir, h))
    if MULTIPROCESSING:
        with Pool(cpu_count()) as p:
            list(tqdm(p.imap(create_mask, args), total=len(args)))
    else:
        for arg in tqdm(args):
            create_mask(arg)

    # Create tracking file
    tracking_file = join(res_dir, "res_track.txt")
    ids = np.unique(h[:, 5]).astype(int)
    births = np.zeros(len(ids))
    ends = np.zeros(len(ids))
    parents = np.zeros(len(ids))
    for i, id in enumerate(ids):
        idx = np.where(h[:, 5] == id)[0]
        births[i] = np.min(h[idx, 0])
        ends[i] = np.max(h[idx, 0])
        parents[i] = h[idx[0], 6]
    parents[parents == -1] = 0
    with open(tracking_file, "w") as f:
        for i, id in enumerate(ids):
            l = f"{int(ids[i])} {int(births[i])} {int(ends[i])}" \
                f" {int(parents[i])}\n"
            f.write(l)
