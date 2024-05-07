import numpy as np
import torch
import torch as t
from datetime import datetime
from torch import nn
import cv2
import io
from torchmetrics import ConfusionMatrix
torch.set_printoptions(precision=2, sci_mode=False, linewidth=1000,
                       edgeitems=100)


class ExtractUncertaintyData:
    """
    Extracts uncertainty data from a segmentation/tracking output of the
    bayesian EmbedTrack neural network.
    """
    def __init__(
            self,
            grid_x: int,
            grid_y: int,
            pixel_x: int,
            pixel_y: int,
            seg: torch.Tensor,
            seg_std: torch.Tensor,
            offset: torch.Tensor = None,
            offset_std: torch.Tensor = None,
            offset_cov_side: torch.Tensor = None,
            seg_cov_side: torch.Tensor = None,
    ):
        self.dev = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        # Create some Grid information of the current image t
        w, h = seg.shape[2], seg.shape[1]
        xm = t.linspace(0, 1, w).view(1, 1, -1).expand(1, h, w)
        ym = t.linspace(0, 1, h).view(1, -1, 1).expand(1, h, w)
        self.yxm = t.cat((ym, xm), 0).to(self.dev)  # (2, y, x)
        self.w, self.h = w, h
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        factor_x = grid_x / (pixel_x * (w - 1))  # Factors to map the grid to
        factor_y = grid_y / (pixel_y * (h - 1))  # the image
        # Output of the EmbedTrack network mapped to the image
        # (EmbedTrack paper for details or in cluster.py def cluster_pixels())
        self.seg = seg.clone()
        self.seg_std = seg_std.clone()
        self.seg_cov_side = seg_cov_side.clone()
        self.seg_min = nn.Sigmoid()(self.seg - self.seg_std)
        self.seg_max = nn.Sigmoid()(self.seg + self.seg_std)
        self.spatial_seed_offset = nn.Tanh()(self.seg[0:2])   # (y, x)
        self.spatial_seed_offset_std = self.seg_std[0:2]
        self.spatial_seed_offset_std[0] *= factor_y
        self.spatial_seed_offset_std[1] *= factor_x
        self.spatial_seed_offset_cov_side = \
            self.seg_cov_side * factor_x * factor_y
        self.spatial_seed_offset[0] *= factor_y
        self.spatial_seed_offset[1] *= factor_x
        self.spatial_seed_offset += self.yxm
        self.sigma = t.exp(nn.Sigmoid()(self.seg[2:4]) * 10)  # (y, x)
        self.seed_map = nn.Sigmoid()(self.seg[4])
        self.seed_map_diff = self.seg_max[4] - self.seg_min[4]

        # Derive the detection certainty from the seed map
        self.detection_certainty = \
            (2*torch.maximum((1 - self.seed_map), self.seed_map)-1).cpu().numpy()
        self.detection_certainty *= 255
        self.detection_certainty = np.round(self.detection_certainty)

        self.probability_of_confusion = 255 - self.detection_certainty

        self.detection_certainty = (1 - self.seed_map_diff.cpu().numpy()) * 255
        self.detection_certainty = np.round(self.detection_certainty)

        # If no offset is given, use a zero tensor instead
        if offset is None:
            offset = torch.zeros((2, h, w)).to(self.dev)
        # If no offset_std is given, use a zero tensor instead
        if offset_std is None:
            offset_std = torch.zeros((2, h, w)).to(self.dev)
        if offset_cov_side is None:
            offset_cov_side = torch.zeros((1, h, w)).to(self.dev)

        # Map the offset prediction to the image
        self.offset = offset.clone()  # (y, x)
        self.offset_std = offset_std.clone()
        self.offset_std[0] *= factor_y
        self.offset_std[1] *= factor_x
        self.offset_cov_side = offset_cov_side.clone() * factor_x * factor_y
        self.mapped_offset = nn.Tanh()(self.offset)  # (y, x)
        self.mapped_offset[0] *= factor_y
        self.mapped_offset[1] *= factor_x
        self.mapped_offset = self.yxm - self.mapped_offset

        # Calculate the +- 1 std offset of the prediction
        #  Tanh can be assumed to be linear in this range close to 0
        self.mapped_offset_min = nn.Tanh()(self.offset - self.offset_std)
        self.mapped_offset_min[0] = self.mapped_offset_min[0] * factor_y
        self.mapped_offset_min[1] = self.mapped_offset_min[1] * factor_x
        self.mapped_offset_min = self.yxm - self.mapped_offset_min
        self.mapped_offset_max = nn.Tanh()(self.offset + self.offset_std)
        self.mapped_offset_max[0] = self.mapped_offset_max[0] * factor_y
        self.mapped_offset_max[1] = self.mapped_offset_max[1] * factor_x
        self.mapped_offset_max = self.yxm - self.mapped_offset_max

        assert torch.all(
            self.offset_std[0] ** 2 *
            self.offset_std[1] ** 2 >=
            self.offset_cov_side ** 2)
        assert torch.all(
            self.spatial_seed_offset_std[0] ** 2 *
            self.spatial_seed_offset_std[1] ** 2 >=
            self.spatial_seed_offset_cov_side ** 2)

        # Check that there is no nan in the input
        assert not torch.isnan(seg).any(), \
            "seg has nan values!"
        assert not torch.isnan(seg_std).any(), \
            "seg_std has nan values!"
        assert not torch.isnan(offset).any(), \
            "offset has nan values!"
        assert not torch.isnan(offset_std).any(), \
            "offset_std has nan values!"
        assert not torch.isnan(offset_cov_side).any(), \
            "offset_cov_side has nan values!"

    @staticmethod
    def merge_gaussian_mixture(labels, w, mu, cov):
        _w, _mu, _cov = [], [], []
        for i in torch.unique(labels):
            if i == -1:
                continue
            inds = labels == i
            w_m = w[inds].sum()
            mu_m = (mu[inds] * w[inds][:, None]).sum(0) / w_m
            diff = mu[inds] - mu_m
            cov_m = (
                (cov[inds] + diff[:, :, None] @ diff[:, None, :]) *
                w[inds][:, None, None]).sum(0) / w_m
            _w.append(w_m)
            _mu.append(mu_m)
            _cov.append(cov_m)
        if len(_w) == 0:
            return torch.zeros((0, 1), device=mu.device), \
                torch.zeros((0, 2), device=mu.device), \
                torch.zeros((0, 2, 2), device=mu.device)
        mu = torch.stack(_mu, 0)
        w = torch.stack(_w)
        cov = torch.stack(_cov)
        return w, mu, cov

    @staticmethod
    def predict_gaussian_classes(
            mu, cov, data_w, data_mu, data_cov, max_mahal=2
    ):
        # Calculate the likelihood of each data point
        if mu.numel() == 0:
            return -torch.ones((data_mu.shape[0]), device=mu.device), \
                   torch.zeros((data_mu.shape[0], 0), device=mu.device)
        l = torch.zeros((data_mu.shape[0], mu.shape[0]), device=mu.device)
        for i in range(mu.shape[0]):
            diff = data_mu - mu[i]
            inv_cov = torch.linalg.inv(cov[i:i + 1] + data_cov)
            mahal_squared = torch.sum(
                (diff[:, None] @ inv_cov)[:, 0] * diff, dim=1)
            mahal = mahal_squared.sqrt()
            l[:, i] = (-0.5 * mahal_squared).exp()
            l[mahal > max_mahal, i] = 0
        l = l / torch.clamp(l.sum(1, keepdims=True), 0.000001)
        l *= data_w[:, None]
        label = torch.argmax(l, 1)
        label[l.max(dim=1)[0] == 0] = -1
        return label, l

    def refine_mask(self, mask: torch.Tensor):
        # Prepare data
        mask = mask.to(self.dev)
        poi = self.seed_map > 0.5
        poi_orig = mask != 0
        w = self.seed_map
        mu = self.spatial_seed_offset.permute(1, 2, 0)
        std = self.spatial_seed_offset_std.permute(1, 2, 0)
        cov_side = self.spatial_seed_offset_cov_side[0]
        mu_old = self.mapped_offset.permute(1, 2, 0)
        std_old = self.offset_std.permute(1, 2, 0)
        cov_side_old = self.offset_cov_side[0]
        cov = torch.zeros((mu.shape[0], mu.shape[1], 2, 2)).to(self.dev)
        cov[:, :, 0, 0] = std[:, :, 0] ** 2
        cov[:, :, 1, 1] = std[:, :, 1] ** 2
        cov[:, :, 0, 1] = cov_side
        cov[:, :, 1, 0] = cov_side
        cov_old = torch.zeros((mu.shape[0], mu.shape[1], 2, 2)).to(self.dev)
        cov_old[:, :, 0, 0] = std_old[:, :, 0] ** 2
        cov_old[:, :, 1, 1] = std_old[:, :, 1] ** 2
        cov_old[:, :, 0, 1] = cov_side_old
        cov_old[:, :, 1, 0] = cov_side_old
        # Extract ROI
        c_w_orig, c_mu_orig, c_cov_orig, label_orig = \
            w[poi_orig], mu[poi_orig], cov[poi_orig], mask[poi_orig]
        c_w_old_orig, c_mu_old_orig, c_cov_old_orig, _ = \
            w[poi_orig], mu_old[poi_orig], cov_old[poi_orig], mask[poi_orig]
        c_w, c_mu, c_cov = w[poi], mu[poi], cov[poi]

        # Refine data mask
        ref_w, ref_mu, ref_cov = self.merge_gaussian_mixture(
            label_orig, c_w_orig, c_mu_orig, c_cov_orig)
        ref_w_old, ref_mu_old, ref_cov_old = self.merge_gaussian_mixture(
            label_orig, c_w_old_orig, c_mu_old_orig, c_cov_old_orig)
        ref_label, ref_likelihood = self.predict_gaussian_classes(
            ref_mu, ref_cov, c_w, c_mu, c_cov)

        # Calc seediness
        probs = torch.zeros(len(ref_w), device=self.dev)
        for l in range(len(ref_w)):
            m = ref_label == l
            if m.sum() == 0:
                continue
            p = torch.max(w[poi][m])  # Changed from mean
            probs[l] = p

        new_mask = torch.zeros_like(mask, device=self.dev)
        new_mask[poi] = ref_label.to(torch.int16) + 1

        return new_mask, ref_mu, ref_cov, ref_mu_old, ref_cov_old, probs

    def get_object_states(self, mask: torch.Tensor):
        """
        Extracts the object states from the input data and stores it to a dict.
        """
        params = dict(
            id=list(),  # Current object center data
            x=list(),
            y=list(),
            cov=list(),
            certainty_probability_of_confusion=list(),  # 1 - clutter intensity
            old_x=list(),  # Tracking offset center data
            old_y=list(),
            old_cov=list(),
            centroid_x=list(),  # Centroid of the object
            centroid_y=list(),
            area=list(),
        )
        ids = t.unique(mask)
        for i in ids:
            if i == 0:
                continue
            m = mask == i
            # Calculate the centroid of the object
            centroid = cv2.moments(m.cpu().numpy().astype(np.uint8))
            centroid_x = centroid["m10"] / centroid["m00"] / self.w
            centroid_y = centroid["m01"] / centroid["m00"] / self.h
            area = m.sum().cpu().item()
            # Calculate the Clutter intensity (probability of confusion)
            seediness = self.seed_map[m]
            certainty_probability_of_confusion = seediness.max()  # Changed
            # from max
            # Calculate the mean and covariance of the spatial seed offset a.k.a
            #  the center of the object
            seed_y = self.spatial_seed_offset[0][m]
            seed_x = self.spatial_seed_offset[1][m]
            s_x = (seed_x * seediness).sum() / seediness.sum()
            s_y = (seed_y * seediness).sum() / seediness.sum()
            # (Create covariances based on the gaussian mixture)
            std_y = self.spatial_seed_offset_std[0][m]
            std_x = self.spatial_seed_offset_std[1][m]
            cov_s = self.spatial_seed_offset_cov_side[0][m]
            cov = torch.zeros((len(std_x), 2, 2), device=self.dev)
            cov[:, 0, 0] = std_x ** 2
            cov[:, 1, 1] = std_y ** 2
            cov[:, 0, 1] = cov_s
            cov[:, 1, 0] = cov_s
            diff_x, diff_y = seed_x - s_x, seed_y - s_y
            cov[:, 0, 0] += diff_x ** 2
            cov[:, 1, 1] += diff_y ** 2
            cov[:, 0, 1] += diff_x * diff_y
            cov[:, 1, 0] += diff_x * diff_y
            cov_new = (cov * seediness[:, None, None]).sum(0) / seediness.sum()
            assert not np.isnan(s_x.cpu().numpy()).any()
            assert not np.isnan(s_y.cpu().numpy()).any()
            # Calculate the mean and covariance of the tracking offset to the
            #   old frame a.k.a the center of the object in the last frame
            old_seed_y = self.mapped_offset[0][m]
            old_seed_x = self.mapped_offset[1][m]
            old_s_x = (old_seed_x * seediness).sum() / seediness.sum()
            old_s_y = (old_seed_y * seediness).sum() / seediness.sum()
            # (Create covariances based on the gaussian mixture)
            std_y = self.offset_std[0][m]
            std_x = self.offset_std[1][m]
            cov_s = self.offset_cov_side[0][m]
            cov_old = torch.zeros((len(std_x), 2, 2), device=self.dev)
            cov_old[:, 0, 0] = std_x ** 2
            cov_old[:, 1, 1] = std_y ** 2
            cov_old[:, 0, 1] = cov_s
            cov_old[:, 1, 0] = cov_s
            diff_x, diff_y = old_seed_x - old_s_x, old_seed_y - old_s_y
            cov_old[:, 0, 0] += diff_x ** 2
            cov_old[:, 1, 1] += diff_y ** 2
            cov_old[:, 0, 1] += diff_x * diff_y
            cov_old[:, 1, 0] += diff_x * diff_y
            cov_old = \
                (cov_old * seediness[:, None, None]).sum(0) / seediness.sum()
            assert torch.all(
                ((cov_old[0,0] * cov_old[1,1]) - (cov_old[0,1]**2)) >= 0), \
                f"Not pos. semi-definit!{cov_old}"
            for _i, x in enumerate([
                i, s_x, s_y, cov_new, certainty_probability_of_confusion,
                old_s_x, old_s_y, cov_old
            ]):
                assert ~np.isnan(x.cpu().numpy()).any(), (x, _i)
            # Add values to parameter dictionary

            params['id'].append(i.cpu().item())
            params['x'].append(s_x.cpu().item())
            params['y'].append(s_y.cpu().item())
            params['cov'].append(cov_new.cpu().numpy())
            params['certainty_probability_of_confusion'].append(
                certainty_probability_of_confusion.cpu().item())
            params['old_x'].append(old_s_x.cpu().item())
            params['old_y'].append(old_s_y.cpu().item())
            params['old_cov'].append(cov_old.cpu().numpy())
            params['centroid_x'].append(centroid_x)
            params['centroid_y'].append(centroid_y)
            params['area'].append(area)
        return params

    @staticmethod
    def add_image_stack_to_params(
            params: dict,
            img_stack: torch.Tensor,
            mus: list,
            covs: list,
            mus_old: list,
            covs_old: list,
            probs: list,
    ):
        """
        Adds the image stack to the parameter dictionary.
        """
        # Compress image stack
        array = img_stack.cpu().numpy()
        compressed_array = io.BytesIO()
        np.savez_compressed(compressed_array, array)
        compressed_array.seek(0)
        params['img_stack'] = compressed_array
        # Switch from yx to xy
        mus = [mu[:, [1, 0]] for mu in mus]
        covs_x = [cov[:, 1, 1].clone() for cov in covs]
        covs_y = [cov[:, 0, 0].clone() for cov in covs]
        for cov, cov_x, cov_y in zip(covs, covs_x, covs_y):
            cov[:, 0, 0] = cov_x
            cov[:, 1, 1] = cov_y
        mus_old = [mu[:, [1, 0]] for mu in mus_old]
        covs_x = [cov[:, 1, 1].clone() for cov in covs_old]
        covs_y = [cov[:, 0, 0].clone() for cov in covs_old]
        for cov, cov_x, cov_y in zip(covs_old, covs_x, covs_y):
            cov[:, 0, 0] = cov_x
            cov[:, 1, 1] = cov_y
        mus = [mu.cpu().numpy() for mu in mus]
        covs = [cov.cpu().numpy() for cov in covs]
        mus_old = [mu.cpu().numpy() for mu in mus_old]
        covs_old = [cov.cpu().numpy() for cov in covs_old]
        # Store
        stack_meta_info = {
            'mus': mus,
            'covs': covs,
            'mus_old': mus_old,
            'covs_old': covs_old,
            'probs': [prob.cpu().numpy() for prob in probs],
        }
        params['img_stack_meta_info'] = stack_meta_info
        return params

    @staticmethod
    def decode_image_stack(params):
        img_stack = np.load(params['img_stack'])
        img_stack.seek(0)
        img_stack = np.load(img_stack)['arr_0']
        return img_stack

    @staticmethod
    def find_error_proposals(
            params: dict,
            inst: torch.Tensor,
            stack: torch.Tensor
    ):
        """
        Finds error proposals in the instance map.
        """
        # Rearange inst labels, such that there are no gaps between the labels
        if type(inst) is np.ndarray:
            inst = torch.from_numpy(inst)
        inst = inst.clone().short()
        stack = stack.clone().short()
        inst = inst.to(stack.device)
        orig_c1 = torch.unique(inst)
        for new, old in enumerate(torch.unique(inst)):
            inst[inst == old] = new
        max_label = inst.max()
        cnf = []
        cnf_inv = []
        orig_c2 = []

        inst_none = inst == 0
        stack_none = stack.max(dim=0)[0] == 0
        all_none = inst_none & stack_none
        finished = False
        for x in range(0, all_none.shape[0]):
            if finished:
                break
            for y in range(0, all_none.shape[1]):
                if all_none[x, y]:
                    all_none[x, y] = 0
                    finished = True
                    break

        inst = inst[~all_none]
        stack = stack[:, ~all_none]

        sus_labels_underseg = []
        sus_labels_overseg = []
        sus_labels_false_pos = []
        sus_labels_false_neg = []

        try:
            for i in range(len(stack)):
                _inst = stack[i].clone()
                orig_c2.append(torch.unique(_inst))
                for new, old in enumerate(torch.unique(_inst)):
                    _inst[stack[i] == old] = new + max_label + 1
                c1 = torch.unique(inst)
                c2 = torch.unique(_inst)
                num_c = len(c1) + len(c2)

                confmat = ConfusionMatrix(task="multiclass", num_classes=int(num_c))
                confmat = confmat.to(inst.device)
                res = confmat(_inst, inst)
                res = res[:len(c1), len(c1):]
                cnf.append(res / res.sum(1, keepdim=True).clamp(0.000001))
                res_inv = confmat(inst, _inst)
                res_inv = res_inv[len(c1):, :len(c1)]
                cnf_inv.append(
                    res_inv / res_inv.sum(1, keepdim=True).clamp(0.000001))

            max_cnf = torch.stack([c[1:, 1:].max(1)[0] for c in cnf], 1)
            max_cnf_min = max_cnf.min(1)[0]
            sus_labels_false_pos = (
                    torch.where(max_cnf_min == 0)[0] + 1
                ).cpu().numpy().tolist()
            max_cnf_inv = [c[1:, 1:].max(1)[0] for c in cnf_inv]
            for f in range(len(cnf_inv)):
                for p in torch.where(max_cnf_inv[f] == 0)[
                             0].cpu().numpy().tolist():
                    sus_labels_false_neg.append((p + 1, f))
            for i in range(len(cnf_inv)):
                c = cnf[i][1:, 1:]
                c_inv = cnf_inv[i][1:, 1:]
                max_c = c.max(1)[0]
                sum_c = c.sum(1)
                max_c_inv = c_inv.max(1)[0]
                sum_c_inv = c_inv.sum(1)

                # Detect potential under segmentations
                sus = ((0.1 < max_c) & (max_c < 0.95)) * (sum_c > 0.95)
                for j in torch.where(sus)[0]:
                    candidates = torch.where(c_inv[:, j] > 0.5)[0]
                    area = c[j, candidates].sum()
                    if area > 0.95:
                        candidates = (candidates + 1).cpu().numpy().tolist()
                        _ = (j.item() + 1, i, candidates)
                        if _ not in sus_labels_underseg:
                            sus_labels_underseg.append(_)

                # Detect potential over segmentations
                sus = ((0.1 < max_c_inv) & (max_c_inv < 0.95)) * (sum_c_inv > 0.95)
                for j in torch.where(sus)[0]:
                    candidates = torch.where(c[:, j] > 0.5)[0]
                    area = c_inv[j, candidates].sum()
                    if area > 0.95:
                        candidates = (candidates + 1).cpu().numpy().tolist()
                        _ = (candidates, i, j.item() + 1)
                        if _ not in sus_labels_underseg:
                            sus_labels_overseg.append(_)

            # Filter the proposals
            _sus_labels_underseg = []
            for j, i, candidates in sus_labels_underseg:
                p = (
                orig_c1[j].item(), i, orig_c2[i][candidates].cpu().numpy().tolist())
                if not any([p[0] == x[0] and len(p[2]) == len(x[2]) for x in
                            _sus_labels_underseg]):
                    _sus_labels_underseg.append(p)
            sus_labels_underseg = _sus_labels_underseg

            _sus_labels_overseg = []
            for candidates, i, j in sus_labels_overseg:
                p = (
                orig_c1[candidates].cpu().numpy().tolist(), i, orig_c2[i][j].item())
                if not any([p[0] == x[0] for x in _sus_labels_overseg]):
                    _sus_labels_overseg.append(p)
            sus_labels_overseg = _sus_labels_overseg

            sus_labels_false_pos = [orig_c1[i].item() for i in sus_labels_false_pos]
        except Exception as e:
            print(e)

        proposals = dict(
            proposals_underseg=sus_labels_underseg,
            proposals_false_pos=sus_labels_false_pos,
            proposals_false_neg=sus_labels_false_neg,
            proposals_overseg=sus_labels_overseg,
        )
        params['error_proposals'] = proposals
        return params
