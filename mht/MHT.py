import cv2
import numpy as np
import datetime
import pickle
from typing import Union
from multiprocessing import Pool, cpu_count
from scipy.optimize import linear_sum_assignment
from scipy.stats import gamma

from mht.utils import Gaussians, \
    PoissonPointProcess as PPP, BernoulliMixture, MultiBernoulliMixture, \
    gaussian_pdf, mahalanobis_distance
from mht.sampling import murty

EPS = 0.000001
np.set_printoptions(precision=3, suppress=True)


class MHTTracker:

    def __init__(
            self,
            debug_mode=False,
            multiprocessing=True,
            mitosis_extension=True,
            mitosis_min_length_a0=None,
            max_number_of_hypotheses=250,
            max_sampling_hypotheses=3,
            gating_probability=0.01,
            gating_distance=10,
            min_sampling_increment=0.01,
            min_object_probability=0.1,
            use_kalman_filter=False,
            P_S=0.9,  # 0.9
            P_B=0.1,  # 0.1
            P_B_border=0.35,
            system_uncertainty=0.0,
    ):
        """
        A PMBM tracker implementation.

        Important notes:
            - The scene is assumed to be a 2D plane and its boarders are defined
                by the x and y axis that are normalized between 0 and 1. The
                origin is in the upper left corner.

        Args:
            debug_mode (bool): If True, the tracker will print debug messages.
            multiprocessing (bool): If True, the tracker will use
                multiprocessing.
            mitosis_extension (bool): If True, the tracker will use the mitosis
                extension.
            mu (float): The mean of the cell lifetime.
            sigma (float): The standard deviation of the cell lifetime.
            mitosis_min_length_a0 (float): The minimum length of a track with
                a parent to have mitosis.
            mitosis_min_length_a (float): The minimum length of a track without
                a parent to have mitosis.
            max_number_of_hypotheses (int): The maximum number of hypotheses
                that will be kept in the PMBM density.
            gating_probability (float): The min probability an association needs
                to have after normalisation to be considered as a valid
                association.
            gating_distance (float): The maximal mahalanobis distance
                (in standard deviations) an association can have (before
                normalizing!) to be considered as a valid association.
            max_sampling_hypotheses (int): The maximum number of hypotheses
                that will be sampled from a hypothesis.
            min_sampling_increment (float): The maximum cumulated probability
                of hypothesis that will be sampled from a hypothesis.
            max_undetected_objects (int): The maximum number of undetected
                objects that will be kept in the PMBM density.
            min_object_probability (float): The minimum probability of an
                undetected object to be kept in the PMBM density.

        Returns:
            None
        """
        # Print input parameters
        print("PMBM tracker parameters:")
        print("    Debug mode: {}".format(debug_mode))
        print("    Multiprocessing: {}".format(multiprocessing))
        print("    Mitosis extension: {}".format(mitosis_extension))
        print("    mitosis_min_length_a0: {}".format(mitosis_min_length_a0))
        print("    max_number_of_hypotheses: {}".format(
            max_number_of_hypotheses))
        print("    max_sampling_hypotheses: {}".format(max_sampling_hypotheses))
        print("    gating_probability: {}".format(gating_probability))
        print("    gating_distance: {}".format(gating_distance))
        print("    min_sampling_increment: {}".format(min_sampling_increment))
        print("    min_object_probability: {}".format(min_object_probability))
        print("    use_kalman_filter: {}".format(use_kalman_filter))
        print("    P_S: {}".format(P_S))
        print("    P_B: {}".format(P_B))
        print("    P_B_border: {}".format(P_B_border))
        print("    system_uncertainty: {}".format(system_uncertainty))

        self.debug_mode = debug_mode
        self.multiprocessing = multiprocessing
        self.mitosis_extension = mitosis_extension
        self.system_uncertainty = system_uncertainty
        self.pool = Pool(cpu_count()) if multiprocessing else None

        """ PMBM density model parameters """
        # Poisson point process density parameters for undetected objects
        self.mbm = MultiBernoulliMixture()
        self.mbm += BernoulliMixture(0, -1, 0)
        self.hypothesis_idx = 1  # Index of the next hypothesis
        self.mitosis_min_length_a0 = mitosis_min_length_a0
        self.P_S = P_S
        self.P_B = P_B
        self.P_B_border = P_B_border

        """ PMBM filter parameters """
        self.max_number_of_hypotheses = max_number_of_hypotheses
        self.gating_probability = gating_probability
        self.gating_distance = gating_distance
        self.max_sampling_hypotheses = max_sampling_hypotheses
        self.min_increment = min_sampling_increment
        self.min_object_probability = min_object_probability
        self.use_kalman_filter = use_kalman_filter

        """ Historical data """
        self.current_frame = 0
        self.association_error = list()
        self.motion = list()
        self.time_stamps = list()
        self.hypotheses = list()
        self.sum_of_probabilities = list()
        self.top_hypothesis = list()
        self.branch_switches = list()
        self.merged_hypotheses = list()

    def step(
            self,
            z: Gaussians,  # Measurements
            z_old: Gaussians,  # Measurement back-projection
            z_is_at_border: np.ndarray,  # Border flag of measurements
            lambda_c_j: np.ndarray,  # Clutter intensity of measurements
            P_D: Union[PPP, np.ndarray],  # Detection intensity
            z_id: np.ndarray = None,  # Indices of measurements
    ):
        """
        Perform one step of the PMBM tracker.

        Usual procedure of performing a step of the PMBM tracker means:
        1. Predicting the next state of objects
        2. Updating the PMBM filter with observations
        3. Reduction of computational complexity
        (4. Extracting the best hypothesis)

        z_pot_overseg: [(z_id1, z_id2),(z_id1, z_id5, z_id8), ...]
        z_pot_underseg: [{
                z_id: id, splits: {z: z, z_old: z_old, lambda_c_j: lambda_c_j}
            }, ...]

        :return:
        # """

        if self.system_uncertainty > 0:
            z_old.P += np.eye(2) * self.system_uncertainty ** 2

        # if self.current_frame > 0:
        #     # Set everything to zero detections
        #     z = Gaussians()
        #     z_old = Gaussians()
        #     z_id = np.zeros(0)
        #     lambda_c_j = np.zeros(0)
        #     z_pot_overseg = list()
        #     z_pot_underseg = list()
        #     z_is_at_border = np.zeros(0, dtype=bool)

        assert ~np.isnan(z.mu).any()
        assert ~np.isnan(z.P).any()
        assert ~np.isnan(z_old.mu).any()
        assert ~np.isnan(z_old.P).any()
        assert ~np.isnan(lambda_c_j).any()
        if P_D is not None:
            if type(P_D) == PPP:
                assert ~np.isnan(P_D.gm.mu).any()
                assert ~np.isnan(P_D.gm.P).any()
                assert ~np.isnan(P_D.w).any()
            elif type(P_D) == np.ndarray:
                assert ~np.isnan(P_D).any()
            else:
                raise TypeError(f"Unknown type of P_D: {type(P_D)}")
        # Associate measurements to background label if no labels are provided
        if self.debug_mode:
            print("\n -------------------")
            print(f"    Hypothesis Likelihoods: "
                  f"{[np.exp(bm.l) for bm in self.mbm.bm]}")
        # Create ids for the measurements if none are provided
        if z_id is None:
            z_id = np.zeros(len(z))
        # Add system noise to the measurements
        if len(self.association_error) > 0:
            measurement_cov = np.eye(2) * np.mean(self.association_error) ** 2
            z_old.P += measurement_cov[None, :, :]
            self.association_error = self.association_error[-10000:]
            self.motion = self.motion[-10000:]
        # Predict and update
        self.predict()
        self.update(
            z=z, z_old=z_old, z_id=z_id, lambda_c_j=lambda_c_j,
            P_D=P_D, z_is_at_border=z_is_at_border
        )
        self.reduce()
        # Update meta data
        for bm in self.mbm.bm:
            bm.age += 1
        self.current_frame += 1
        # Store historical data
        self.time_stamps.append(datetime.datetime.now())
        hypothesis = list()
        for bm in self.mbm.bm:
            hypothesis.append({
                "id": bm.identifier, "parent": bm.parent, "l": bm.l,
                "r": np.copy(bm.r), "mu": np.copy(bm.gm.mu),
                "P": np.copy(bm.gm.P), "parent_label": np.copy(bm.parent_label),
                "age": np.copy(bm.age), "labels": np.copy(bm.labels),
                "associated_id": np.copy(bm.associated_id),
            })
        self.hypotheses.append(hypothesis)
        if self.top_hypothesis:
            if self.mbm.bm[0].parent != self.top_hypothesis[-1]:
                self.branch_switches.append(self.current_frame)
                if self.debug_mode:
                    print("    Branch switch between: ", self.mbm.bm[0].parent,
                          self.top_hypothesis[-1])
        self.top_hypothesis.append(self.mbm.bm[0].identifier)
        # Prune unnecessary historical data
        if (self.current_frame % 20) == 0:
            self.clear_history()

    ''' Prediction '''
    def predict(self):
        """ Predict the next state of objects"""
        # Predict the new state density ...
        motion = np.mean(self.motion) \
            if len(self.motion) > 0 else 0.01
        self.motion = self.motion[-1000:]
        Q_mov = np.eye(2) * motion ** 2
        # ... of detected objects
        #   --> Does not need to be performed in our tracker!
        for i in range(len(self.mbm.bm)):
            self.mbm.bm[i].r *= self.P_S
        if self.use_kalman_filter:
            for i in range(len(self.mbm.bm)):
                self.mbm.bm[i].gm.P += Q_mov[None, ]

    ''' Update '''
    def update(
            self,
            z: Gaussians,
            z_old: Gaussians,
            z_id: np.ndarray,
            z_is_at_border: np.ndarray,
            lambda_c_j: np.ndarray,
            P_D: Union[PPP, np.ndarray],
    ):
        """ Update the PMBM filter with the given measurements. """
        assert len(z) == len(z_old) == len(lambda_c_j)
        self.debug(f"--- Updating the PMBM filter with measurements ---")
        self.debug(f"    Number of measurements: {len(z)}")
        self.debug(f"    Associated IDS (z_id): {z_id}")
        self.debug(f"    Clutter prob (lambda_c_j): {lambda_c_j}")

        ''' Set the movement to 0 if using Kalman filter '''
        if self.use_kalman_filter:
            motion = np.mean(self.motion) if len(self.motion) > 0 else 0
            Q_mov = np.eye(2) * motion ** 2
            z_old.mu = np.copy(z.mu)
            z.P = np.zeros_like(z.P)
            z_old.P = np.zeros_like(z.P) + Q_mov[None, :, :]

        ''' Create new Bernoullis for potentially new objects 
        (This need to be done before updating the Bernoulli mixtures of
        undetected objects, because the new Bernoullis might be detected
        objects) 
        '''
        p_b = self.P_B if self.current_frame > 0 else .95
        p_b_border = self.P_B_border if self.current_frame > 0 else .95
        s_jt_num = self._create_s_jt_numerator(
            p_b, p_b_border, z_is_at_border, lambda_c_j)

        # Set associated ids to 0
        for bm in self.mbm.bm:
            bm.associated_id = np.zeros_like(bm.associated_id)

        ''' Create new hypotheses based on the measurements '''
        # Iterate over all Bernoulli mixtures
        args = []
        max_motion = np.mean(self.motion) if len(self.motion) > 0 else 0
        Q_mov = np.eye(2) * max_motion ** 2

        for h in range(len(self.mbm)):
            bm = self.mbm.bm[h]
            debug = self.debug_mode and h == 0 and i == 0
            args.append(dict(
                bm=bm, z=z, z_old=z_old, z_id=z_id, s_jt_num=s_jt_num,
                lambda_c_j=lambda_c_j, P_D=P_D,
                gating_distance=self.gating_distance,
                gating_probability=self.gating_probability,
                max_sampling_hypotheses=self.max_sampling_hypotheses,
                hypothesis_idx=self.hypothesis_idx, debug=debug,
                min_increment=self.min_increment,
                mitosis_min_length_a0=self.mitosis_min_length_a0,
                max_motion=max_motion,
                use_kalman_filter=self.use_kalman_filter,
                Q_mov=Q_mov,
            ))
            self.hypothesis_idx += self.max_sampling_hypotheses

        new_hypotheses = []
        association_error = []
        motion = []
        if self.multiprocessing:
            args = [(self._deduce_hypothesis, arg) for arg in args]
            # if len(self.branch_switches) > 1:
            #     args = args[0:1]
            _new_hypothesis = self.pool.starmap(self.dict_wrapper, args)
            if debug:
                print(" Max. sampled hypothesis ",
                      max([len(h) for h, _, _ in _new_hypothesis]))
            for h, _association_error, _motion in _new_hypothesis:
                new_hypotheses += h
                association_error += _association_error
                motion += _motion
        else:
            for arg in args:
                _new_hypothesis, _association_error, _motion = \
                    self._deduce_hypothesis(**arg)
                new_hypotheses += _new_hypothesis
                association_error += _association_error
                motion += _motion
        self.association_error += association_error[0]
        motion = np.array(motion[0])  # Remove motion of rigid noisy objects
        self.motion += motion[motion > 0.001].tolist()

        ''' Add hypotheses to the new MBM '''
        self.mbm = MultiBernoulliMixture()
        for h in new_hypotheses:
            assert not np.isnan(h.l).any(), f"h has nan weight: {h.l}"
            self.mbm += h
        if self.debug_mode:
            print(f"    Max Hypotheses Weights: "
                  f"{max([bm.l for bm in self.mbm.bm])}")
            print(f"    Hypotheses Weights: "
                  f"{[int(100 * bm.l) / 100 for bm in self.mbm.bm]}")

        ''' Normalize the log weights of the hypotheses '''
        self.mbm.normalize()

    @staticmethod
    def _deduce_hypothesis(
            bm: BernoulliMixture,
            z: Gaussians,
            z_old: Gaussians,
            z_id: np.ndarray,
            s_jt_num: np.ndarray,
            lambda_c_j: np.ndarray,
            P_D: Union[PPP, np.ndarray],
            gating_distance: float,
            gating_probability: float,
            max_sampling_hypotheses: int,
            min_increment: float,
            hypothesis_idx,
            debug: bool = False,
            mitosis_min_length_a0: int = None,
            max_motion: float = None,
            use_kalman_filter: bool = False,
            Q_mov: np.ndarray = None,
            likelihood: float = 1,
    ):
        """ Create new hypotheses based on the measurements """
        # Calculate the probability of detection fot each object
        bm.l = np.log(likelihood) + bm.l
        P_D_i = MHTTracker._create_P_D_i(P_D=P_D, bm=bm)
        # Calculate the expectation of a cell division/mitosis
        pt_i = MHTTracker._create_pt_i(
            age=bm.age, has_parent=bm.parent_label > 0,
            min_length_a0=mitosis_min_length_a0,
        )
        # Find close neighbours
        max_motion = 0 if max_motion is None else max_motion * 10
        if debug:
            print(f"    max_motion: {max_motion}")
        # Get Normalized s_ij and s_jt
        s_ij_num = MHTTracker._create_s_ij_num(
            bm=bm, z_old=z_old, P_D_i=P_D_i, lambda_c_j=lambda_c_j,
            gate=gating_distance,
        )
        s_ij, s_jt = MHTTracker._normalize_s_ij_s_jt(
            s_ij_num=s_ij_num, s_jt_num=s_jt_num, lambda_c_j=lambda_c_j,
        )
        # Create pdf and logits for new objects
        uj_pdf = MHTTracker._create_pdf_j0(
            z=z, lambda_c_j=lambda_c_j, s_jt=s_jt
        )
        l_uj = MHTTracker._create_l_j0(lambda_c_j=lambda_c_j, s_jt=s_jt)
        # Create probability of mis-detection of old objects
        l_i0, r_MD = MHTTracker._create_l_i0(bm=bm, P_D_i=P_D_i)
        # Create probability of association between objects and measurements
        l_ij = MHTTracker._create_l_ij(bm=bm, P_D_i=P_D_i, s_ij=s_ij)
        costs = MHTTracker._create_cost_matrix(
            l_uj=l_uj, s_ij=s_ij, pt_i=pt_i, min_probability=gating_probability,
        )
        # Sampling to from M best hypotheses
        hypotheses, likelihoods = murty(
            costs=costs,
            max_hypotheses=max_sampling_hypotheses,
            min_increment=min_increment,
        )
        # assert np.argsort([-l for l in likelihoods])[0] == 0, \
        #    f"The best hypothesis is not the first one {likelihoods}"
        # --> Can be caused by numerical instabilities...
        if debug:
            print(f"    (Hypothesis 1)")
            print(f"    bm.labels: {bm.labels.squeeze()}")
            print(f"    bm.r: {bm.r.squeeze()}")
            print(f"    P_D_i: {P_D_i.squeeze()}")
            print(f"    l_ij:\n")
            for i, _l in enumerate(bm.labels.squeeze()):
                print(f"        {_l}: \t{l_ij.squeeze()[i]}")
            print(f"    l_i0: {l_i0.squeeze()}")
            print(f"    l_uj: {l_uj.squeeze()}")
            print(f"    uj_pdf.w: {uj_pdf.w.squeeze()}")
            #print(f"    cost_matrix: \n{costs.squeeze()}")
            #print(f"    hypothesis likelihoods: {likelihoods}")
        # Create new hypotheses
        assert len(hypotheses) > 0, "No hypotheses found"
        new_hypotheses = []
        association_error = list()
        motion = list()
        for i, hypothesis in enumerate(hypotheses):
            _debug = debug and i == 0
            new_h, _association_error, _motion = MHTTracker._create_hypothesis(
                bm=bm, new_obj_pdf=uj_pdf, l_uj=l_uj, l_ij=l_ij, l_i0=l_i0,
                hypothesis=hypothesis, hypothesis_idx=hypothesis_idx + i,
                z=z, z_old=z_old, z_id=z_id, r_MD=r_MD,
                debug=_debug, use_kalman_filter=use_kalman_filter,
                Q_mov=Q_mov,
            )
            new_hypotheses.append(new_h)
            if i == 0:
                association_error = _association_error
                motion = _motion
        return new_hypotheses, association_error, motion

    @staticmethod
    def _create_P_D_i(
            P_D: Union[PPP, np.ndarray],
            bm: BernoulliMixture
    ):
        if type(P_D) == PPP:
            P_D_i = P_D.all_probabilities((bm.gm.mu))
        elif type(P_D) == np.ndarray:
            P_D_i = np.zeros((len(bm), 1))
            H, W = P_D.shape[0:2]
            for i in range(len(bm)):
                x = int(np.clip(bm.gm.mu[i, 0] * W, 0, W - 1))
                y = int(np.clip(bm.gm.mu[i, 1] * H, 0, H - 1))
                # cv2.imshow("P_D", P_D)
                # cv2.waitKey(0)
                P_D_i[i] = P_D[y, x]
        else:
            raise TypeError(f"Unknown type of P_D: {type(P_D)}")
        P_D_i = np.clip(P_D_i, 0, 1)[:, 0]
        return P_D_i

    @staticmethod
    def _create_pt_i(
            age: np.ndarray,
            has_parent: np.ndarray,
            min_length_a0: int,
    ) -> np.ndarray:
        """ Create the probability of a cell division/mitosis. """
        # Create the probability of a cell division/mitosis for specific ages
        pt_i = np.ones(len(age))
        has_parent = has_parent[:, 0]
        probs_a0 = np.ones(5000)
        probs_a = np.ones(5000)
        # Apply threshold or let probabilities be 1
        if min_length_a0 is not None:
            probs_a0[0:min_length_a0] = 0
            ### Linear Ramp
            # probs_a0[0:min_length_a0] = np.arange(min_length_a0) / min_length_a0
            ### Erlang distributed
            mu, sigma = min_length_a0, min_length_a0
            alpha = mu ** 2 / sigma ** 2
            beta = mu / sigma ** 2
            erl = gamma(a=alpha, scale=1 / beta)
            probs_a0 = erl.cdf(np.linspace(0, 5000, 5000))
            ### Clip at 5%
            probs_a0 = np.maximum(0.05, probs_a0)
        # Complete tracks
        pt_i[has_parent] = probs_a0[age[has_parent].astype(int).squeeze()]
        # Incomplete tracks
        pt_i[~has_parent] = probs_a[age[~has_parent].astype(int).squeeze()]
        return pt_i

    @staticmethod
    def _create_s_jt_numerator(
            P_B: float,
            P_B_border: float,
            is_j_at_border: np.ndarray,
            lambda_c_j: np.ndarray,
    ) -> (np.ndarray, PPP):
        """ Create new Bernoullis for potentially new objects. """
        assert len(lambda_c_j) == len(is_j_at_border), \
            (len(lambda_c_j), len(is_j_at_border))
        birth_probability = np.ones(len(lambda_c_j)) * P_B
        birth_probability[is_j_at_border] = P_B_border
        s_jt_num = birth_probability * (1 - lambda_c_j[:, None])
        assert ~np.isnan(s_jt_num).any(), s_jt_num
        return s_jt_num

    @staticmethod
    def _create_s_ij_num(
            bm: BernoulliMixture,
            z_old: Gaussians,
            P_D_i: np.ndarray,
            lambda_c_j: np.ndarray,
            gate: float,
    ) -> (np.ndarray, np.ndarray):
        # Create probability of association between objects and measurements
        assert len(bm) == len(P_D_i)
        assert P_D_i.shape == (len(bm),)
        # Expand matrices for vectorized operations
        objects = bm.__copy__()
        obj_mu = np.repeat(objects.gm.mu[:, None, ], len(z_old), axis=1)
        obj_P = np.repeat(objects.gm.P[:, None, ], len(z_old), axis=1)
        obj_r = np.repeat(objects.r[:, None, ], len(z_old), axis=1)[:, :, 0]
        z_old_mu = np.repeat(z_old.mu[None,], len(objects), axis=0)
        z_old_P = np.repeat(z_old.P[None,], len(objects), axis=0)
        # Calculate the mahalanobis distance between objects and measurements
        mahal = mahalanobis_distance(obj_mu, z_old_mu, obj_P, z_old_P)
        # Calculate the probability of association
        s_ij_num = gaussian_pdf(z_old_mu, obj_mu, obj_P + z_old_P)
        s_ij_num[mahal > gate] = 0
        s_ij_num *= obj_r * (1 - lambda_c_j[None, :])
        assert ~np.isnan(s_ij_num).any(), s_ij_num
        return s_ij_num

    @staticmethod
    def _normalize_s_ij_s_jt(
            s_ij_num: np.ndarray,
            s_jt_num: np.ndarray,
            lambda_c_j: np.ndarray,
    ):
        """ Normalize the probabilities of existence. """
        assert s_ij_num.shape[1] == s_jt_num.shape[0], \
            (s_ij_num.shape[1], s_jt_num.shape[0])
        assert len(lambda_c_j) == s_jt_num.shape[0]
        # Normalize without neighbours
        denominator = s_ij_num.sum(axis=0) + s_jt_num.sum(axis=1)
        s_ij = s_ij_num * (1 - lambda_c_j[None, :]) /\
               np.maximum(denominator[None, :], EPS)
        s_jt = s_jt_num * (1 - lambda_c_j[:, None]) /\
               np.maximum(denominator[:, None], EPS)
        s_ij = np.minimum(s_ij, .999999)
        assert ~np.isnan(s_ij).any(), s_ij
        assert ~np.isnan(s_jt).any(), s_jt
        return s_ij, s_jt

    @staticmethod
    def _create_pdf_j0(
            z: Gaussians,
            lambda_c_j: np.ndarray,
            s_jt: np.ndarray,
    ) -> (np.ndarray, PPP):
        """ Create new Bernoullis for potentially new objects. """
        assert len(z) == len(lambda_c_j)
        assert lambda_c_j.shape == (len(z), )
        rho_z = s_jt / np.maximum((1 - lambda_c_j[:, None]), EPS)
        rho_z = np.sum(rho_z, axis=1, keepdims=True)
        r_i = rho_z / (lambda_c_j[:, None] + rho_z)
        pdfs = PPP(w=r_i, gm=Gaussians(z.mu, z.P))
        return pdfs

    @staticmethod
    def _create_l_j0(
            lambda_c_j: np.ndarray,
            s_jt: np.ndarray,
    ) -> (np.ndarray, PPP):
        """ Create new Bernoullis for potentially new objects. """
        assert lambda_c_j.shape == (len(s_jt), )
        # Calculate the log likelihood for every new object
        rho_z = s_jt / np.maximum((1 - lambda_c_j[:, None]), EPS)
        rho_z = np.sum(rho_z, axis=1, keepdims=True)
        p_uj = rho_z + lambda_c_j[:, None]
        l_uj = np.log(np.clip(p_uj, EPS, 1 - EPS))[:, 0]
        assert not np.isnan(l_uj).any(), f"nan in l_uj: {l_uj}"
        return l_uj

    @staticmethod
    def _create_l_ij(
            bm: BernoulliMixture,
            P_D_i: np.ndarray,
            s_ij: np.ndarray,
    ) -> (np.ndarray, np.ndarray):
        # Create probability of association between objects and measurements
        assert len(bm) == len(P_D_i)
        assert P_D_i.shape == (len(bm),)
        objects = bm.__copy__()
        I, J = s_ij.shape[0:2]
        obj_r = np.repeat(objects.r[:, None, ], J, axis=1)[:, :, 0]
        with np.errstate(divide='ignore'):
            l_ij = np.log(s_ij) + np.log(obj_r * P_D_i[:, None])
        assert not np.isnan(l_ij).any(), f"nan in l_uj: {l_ij}"
        return l_ij

    @staticmethod
    def _create_l_i0(
            bm: BernoulliMixture,
            P_D_i: np.ndarray,
    ) -> (np.ndarray, PPP):
        assert len(bm) == len(P_D_i), (len(bm), len(P_D_i))
        assert P_D_i.shape == (len(bm),)
        r_old = np.copy(bm.r)[:, 0]
        P_MD = 1 - P_D_i
        p_i0 = 1 - r_old * P_D_i
        l_i0 = np.log(np.clip(p_i0, EPS, 1 - EPS))
        r_MD = r_old * P_MD / p_i0

        assert (0 <= r_MD).all() and (r_MD <= 1).all(), f"r_MD : {r_MD}"
        assert not np.isnan(l_i0).any(), f"nan in l_i0: {l_i0}"
        assert not np.isnan(r_MD).any(), f"nan in r_MD: {r_MD}"
        return l_i0, r_MD

    @staticmethod
    def _create_cost_matrix(
            l_uj: np.ndarray,
            s_ij: np.ndarray,
            pt_i: np.ndarray,
            min_probability: float,
    ) -> np.ndarray:
        assert l_uj.shape == (s_ij.shape[1], ), (s_ij.shape, l_uj.shape)
        # Create probability of association between objects and measurements
        I, J = s_ij.shape
        costs = -np.ones((J, 2 * I + J)) * np.inf
        with np.errstate(divide='ignore'):
            costs[:J, :I] = (np.log(s_ij)).transpose()
        _l_uj = np.eye(J)
        _l_uj[_l_uj == 1] = l_uj
        _l_uj[_l_uj == 0] = -np.inf
        costs[:J, 2*I:] = _l_uj
        # Normalize the log weights
        p = np.exp(costs)
        assert not np.isnan(p).any(), f"nan in p: {p}"
        # Gate unlikely measurements
        p[:J, :2*I] = (p[:J, :2*I] > min_probability) * p[:J, :2*I]
        with np.errstate(divide='ignore'):
            costs = np.log(p)
        # NOTE: The next step is necessary to assign first measurement j1 < j2
        #   that are assigned to the same object to the first cell split area
        #   to avoid issues with the Hungarian algorithm
        epsilon = (J - np.arange(J)) * EPS
        costs[:J, I:2*I] = np.copy(costs[:J, :I]) - epsilon[:, None]
        # Add costs for cell split
        with np.errstate(divide='ignore'):
            split_costs = np.log(pt_i[None, :])
        costs[:J, I:2 * I] += split_costs
        assert not np.isnan(epsilon).any(), f"nan in epsilon: {epsilon}"

        return -costs

    @staticmethod
    def _create_hypothesis(
            hypothesis: tuple,
            hypothesis_idx: int,
            bm: BernoulliMixture,
            z: Gaussians,
            z_old: Gaussians,
            z_id: np.ndarray,
            r_MD: np.ndarray,
            new_obj_pdf: PPP,
            l_uj: np.ndarray,
            l_ij: np.ndarray,
            l_i0: np.ndarray,
            debug=False,
            use_kalman_filter=False,
            Q_mov=None,
    ):
        # Create new hypotheses
        rows, cols = hypothesis
        assert rows.size == np.unique(rows).size, f"rows: {rows}"
        new_h = bm.get_child(new_identifier=hypothesis_idx)
        assert not np.isnan(new_h.l)
        if debug:
            print("------")
            print(f"Old likelihood {bm.l}")
            print(f"    ids i={np.arange(0, len(bm))}: {bm.labels.squeeze()}")
        log_likelihood = 0
        # Log association error and motion
        association_error = list()
        motion = list()
        # Update objects with detections which have ...
        associated = np.isin(np.arange(len(bm)), cols) | \
                     np.isin(np.arange(len(bm)) + len(bm.gm), cols)
        split = np.isin(np.arange(len(bm)), cols) & \
                np.isin(np.arange(len(bm)) + len(bm.gm), cols)
        single_associated = associated & ~split
        # ... a mis-detection
        new_h.r[~associated] = r_MD[~associated, None]
        if debug:
            print("rmd", r_MD)
            print("~assoc", ~associated)
            print("r", new_h.r[:, 0])
        new_h.gm.P[~associated] += Q_mov[None, ]
        if debug:
            print(f"    l_i0 i={np.where(~associated)[0]}: {l_i0[~associated]}")
        log_likelihood += np.sum(l_i0[~associated])
        # ... an associated measurement
        i = np.where(single_associated)[0]
        j = [rows[list(cols).index(x)] for x in i]
        _association_error = np.sum((new_h.gm.mu[i] - z_old.mu[j]) ** 2, axis=1)
        association_error.append(np.sqrt(_association_error).tolist())
        _motion = np.sum((new_h.gm.mu[i] - z.mu[j]) ** 2, axis=1)
        motion.append(np.sqrt(_motion).tolist())
        if use_kalman_filter:
            K = new_h.gm.P[i] @ np.linalg.inv(new_h.gm.P[i] + z.P[j])
            new_h.gm.mu[i] += \
                (K @ (z.mu[j] - new_h.gm.mu[i])[..., None])[..., 0]
            new_h.gm.P[i] -= K @ new_h.gm.P[i]
        else:
            new_h.gm.mu[i] = z.mu[j]
            new_h.gm.P[i] = z.P[j]
        new_h.r[i] = 1
        if debug:
            print(new_h.r[:, 0], i)
        new_h.associated_id[i] = z_id[j, None]
        if debug:
            print(f"    l_ij i={i}, j={j},  assoc_id={z_id[j]}: {l_ij[i, j]}")
        log_likelihood += np.sum(l_ij[i, j])
        # ... have two associated measurements (Cell Split)
        i = np.where(split)[0]
        j1 = [rows[list(cols).index(x)] for x in i]
        j2 = [rows[list(cols).index(x + len(bm.gm))] for x in i]
        if use_kalman_filter:
            K1 = new_h.gm.P[i] @ np.linalg.inv(new_h.gm.P[i] + z.P[j1])
            K2 = new_h.gm.P[i] @ np.linalg.inv(new_h.gm.P[i] + z.P[j2])
            new_mu1 = new_h.gm.mu[i] + (K1 @
                       (z.mu[j1] - new_h.gm.mu[i])[..., None])[..., 0]
            new_P1 = new_h.gm.P[i] - K1 @ new_h.gm.P[i]
            new_mu2 = new_h.gm.mu[i] + (K2 @
                       (z.mu[j2] - new_h.gm.mu[i])[..., None])[..., 0]
            new_P2 = new_h.gm.P[i] - K2 @ new_h.gm.P[i]
        else:
            new_mu1 = z.mu[j1]
            new_mu2 = z.mu[j2]
            new_P1 = z.P[j1]
            new_P2 = z.P[j2]
        for x, _i in enumerate(i):  # Add new objects
            new_h += ([[1.0]], Gaussians([new_mu1[x]], [new_P1[x]]))
            new_h.parent_label[-1] = bm.labels[_i]
            new_h.associated_id[-1] = z_id[j1[x], None]
            new_h += ([[1.0]], Gaussians([new_mu2[x]], [new_P2[x]]))
            new_h.parent_label[-1] = bm.labels[_i]
            new_h.associated_id[-1] = z_id[j2[x], None]
        for x, _i in enumerate(reversed(i)):  # Remove old objects
            if new_h.parent_label[_i] >= 0:
                new_h.complete_track_lengths.append(
                    (int(new_h.labels[_i]), int(new_h.age[_i])))
            else:
                new_h.incomplete_track_lengths.append(
                    (int(new_h.labels[_i]), int(new_h.age[_i])))
            new_h.remove(_i)
        if debug:
            print(f"    l_ij i={i}, j1={j1}: {l_ij[i, j1]}")
            print(f"    l_ij i={i}, j2={j2}: {l_ij[i, j2]}")
        log_likelihood += np.sum(l_ij[i, j1] + l_ij[i, j2])
        # Create new objects for measurements which have no association
        unassociated = np.where(cols >= 2 * len(bm))[0]
        for j in unassociated:
            new_h += new_obj_pdf[j]
            new_h.associated_id[-1] = z_id[j, None]
            if debug:
                print(f"    l_uj j={j}: {l_uj[j]}")
            log_likelihood += l_uj[j]
        # Update log likelihood
        new_h.l += float(log_likelihood)
        if debug:
            print(f"New likelihood {new_h.l}")
        assert not np.isnan(new_h.l).any(), 'Log likelihood is NaN!'
        return new_h, association_error, motion

    ''' Reduce complexity of the PMBM '''
    def reduce(self):
        self._merge()
        self._capping()
        self._prune()
        self._recycling()

    def _prune(self):
        # Remove, such that only the best hypotheses that cover 99% of the
        #   total likelihood are kept
        self.mbm.normalize()
        self.mbm.prune(0.95)
        self.mbm.normalize()

    def _merge(self):
        # Sort all objects based on their positions
        self.mbm.normalize()
        num_hypotheses_before_merging = len(self.mbm)
        num_objects, positions, ages = list(), list(), list()
        sum_positions, sum_ages = list(), list()
        for h in self.mbm:
            num_objects.append(len(h.gm))
            _position = h.gm.mu[:, 0] * 100 + h.gm.mu[:, 1]
            inds = np.argsort(_position)
            _position = _position[inds]
            _age = h.age[inds, 0]
            _label = h.labels[inds]
            _unique_label, _label_counts = np.unique(_label, return_counts=True)
            # if self.mitosis_min_length_a0:
            #     _age[_age > self.mitosis_min_length_a0] = self.mitosis_min_length_a0
            if self.mitosis_min_length_a0 is None:
                _age *= 0
            positions.append(_position)
            ages.append(_age)
            sum_positions.append(np.sum(_position))
            sum_ages.append(np.sum(h.age[:, 0]))
        # Find same hypothesis
        tested_hypotheses = list()
        same_hypotheses = list()
        if self.debug_mode:
            print(f"Merged Hypothesis: {same_hypotheses}" )
        for h, (n, pos, age, s_pos, s_age) in enumerate(zip(
            num_objects, positions, ages, sum_positions, sum_ages)
        ):
            if h in tested_hypotheses:
                continue
            tested_hypotheses.append(h)
            pot_hypotheses = np.argwhere(
                (np.asarray(num_objects) == n) *
                (np.asarray(sum_positions) == s_pos) *
                (np.asarray(sum_ages) == s_age)
            )
            pot_hypotheses = np.setdiff1d(pot_hypotheses, tested_hypotheses)
            if pot_hypotheses.size == 0:
                same_hypotheses.append([h])
                continue
            # Check if ages are the same
            _ages = np.stack([ages[p] for p in pot_hypotheses])
            _ages_diff = np.abs(_ages - age[None, :]).sum(axis=1)
            _same_age = np.argwhere(_ages_diff == 0)
            if _same_age.size == 0:
                same_hypotheses.append([h])
                continue
            pot_hypotheses = pot_hypotheses[_same_age[:, 0]]
            # Check if positions are the same
            _positions = np.stack([positions[p] for p in pot_hypotheses])
            _positions_diff = np.abs(_positions - pos[None, :]).sum(axis=1)
            _same_position = np.argwhere(_positions_diff == 0)
            if _same_position.size == 0:
                same_hypotheses.append([h])
                continue
            pot_hypotheses = pot_hypotheses[_same_position[:, 0]]
            # Remove the hypothesis from the search space
            tested_hypotheses.extend(pot_hypotheses.tolist())
            same_hypotheses.append([h] + pot_hypotheses.tolist())
        # Merge the hypotheses
        new_mbm = list()
        for h_set in same_hypotheses:
            likelihoods = np.asarray([np.exp(self.mbm[h].l) for h in h_set])
            max_likelihood = np.argmax(likelihoods)
            new_mbm.append(self.mbm[h_set[max_likelihood]])
            if len(h_set) == 1:
                continue
            #new_mbm[-1].l = np.log(np.sum(likelihoods))
        self.mbm.bm = new_mbm
        self.mbm.normalize()
        num_hypotheses_after_merging = len(self.mbm)
        self.merged_hypotheses.append(
            (num_hypotheses_before_merging, num_hypotheses_after_merging)
        )

    def _capping(self):
        self.mbm.keep_top_k(self.max_number_of_hypotheses)
        total_probability = self.mbm.get_total_probability()
        self.mbm.normalize()
        self.sum_of_probabilities.append(total_probability)

    def _recycling(self):
        for h in range(len(self.mbm)):
            objects_to_remove = []
            for i in range(len(self.mbm[h].gm)):
                if self.mbm[h].r[i] < self.min_object_probability:
                    w = self.mbm[h].r[i] * np.exp(self.mbm[h].l)
                    assert not np.isnan(w).any(), \
                        f'w is NaN! r={self.mbm[h].r[i]} l={self.mbm[h].l}'
                    objects_to_remove.append(i)
            for i in reversed(objects_to_remove):
                self.mbm[h].remove(i)

    ''' Utility functions '''
    def get_historical_data(self):
        historic_data = {
            'time_stamps': self.time_stamps,
            'hypotheses': self.hypotheses,
            'debug_mode': self.debug_mode,
            'processing_time': self.time_stamps[-1] - self.time_stamps[0],
        }
        return historic_data

    def clear_history(self):
        pass

    def store(self, path):
        historic_data = self.get_historical_data()
        with open(path, 'wb') as f:
            pickle.dump(historic_data, f)

    def estimate(self, prune=True):
        bm = self.mbm.get_best_hypothesis()
        bm = bm.__copy__()
        if prune:
            bm.prune(0.5)
        return bm

    def __str__(self):
        processing_time = self.time_stamps[-1] - self.time_stamps[0]
        s = f"PMBMTracker("
        s += f"hypotheses={len(self.mbm)}, "
        s += f"top_likelihood={self.mbm.top_likelihood()}, "
        s += f"branch_sw={len(self.branch_switches)}, "
        s += f"processing_time={processing_time}, "
        s += ")"
        return s

    def debug(self, s):
        if self.debug_mode:
            print(s)

    @staticmethod
    def dict_wrapper(f, dict_args):
        return f(**dict_args)


if __name__ == '__main__':
    from pathlib import Path
    import PMBM.dataloader as dl

    data = dl.CellTrackingChallengeSequence(
        path=Path(r"C:\Users\kaiser\code\CellTrackingChallenge\data\challenge"),
        dataset_name="BF-C2DL-HSC",
        sequence_name="01"
    )
    tracker = MHTTracker()

    for i in range(len(data)):
        d = data[i]

        tracker.step(
            z=d["z"],
            z_old=d["z_old"],
            lambda_c_j=d["lambda_c_j"],
            P_B=d["P_B"],
            P_D=d["P_D"],
        )
        print(f"(Frame {i}) {tracker}", flush=True)


