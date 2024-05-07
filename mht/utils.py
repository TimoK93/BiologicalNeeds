import numpy as np
from copy import deepcopy
from numpy import vstack, maximum, sign, concatenate, pi, log, exp, sum
from numpy.linalg import cholesky, det, inv


def gaussian_pdf(x, mu, P):
    assert x.ndim == 2 or mu.ndim == 3, x.ndim
    assert x.shape == mu.shape, (x.shape, mu.shape)
    assert P.shape[-1] == P.shape[-2] == 2, P.shape
    assert mu.shape[-1] == 2, mu.shape
    ori_dims = x.shape
    if len(ori_dims) == 3:
        x = x.reshape((-1, 2))
        mu = mu.reshape((-1, 2))
        P = P.reshape((-1, 2, 2))
    # Scale to be more stable
    s = 1000
    x, mu, P = x * s, mu * s, P * s ** 2
    # Thresholding to be numerically stable
    e = 0.000001
    P[:, 0, 0] = maximum(P[:, 0, 0], 10 * e)
    P[:, 1, 1] = maximum(P[:, 1, 1], 10 * e)
    P[:, 0, 1] = sign(P[:, 0, 1]) * maximum(np.abs(P[:, 0, 1]), e)
    P[:, 1, 0] = sign(P[:, 1, 0]) * maximum(np.abs(P[:, 1, 0]), e)
    diff = x - mu
    # Stable version of multivariate normal distribution
    # Calculate determinant of via Choelsky decomposition
    chol = cholesky(2 * pi * P)
    determinant = det(chol) * det(np.transpose(chol, (0, 2, 1)))
    log_like = -1 / 2 * log(determinant)[:, None]
    log_like -= (diff[:, None] @ inv(P) @ diff[:, :, None])[:, :, 0] / 2
    ret = exp(log_like)
    ret *= s ** 2
    if len(ori_dims) == 3:
        ret = ret.reshape((ori_dims[0], ori_dims[1]))
    return ret


def mahalanobis_distance(mu1, mu2, P1, P2):
    assert mu1.ndim == 2 or mu1.ndim == 3, mu1.ndim
    assert mu1.shape == mu2.shape, (mu1.shape, mu2.shape)
    assert P1.shape[-1] == P1.shape[-2] == 2, P1.shape
    assert P2.shape[-1] == P2.shape[-2] == 2, P2.shape
    ori_dims = mu1.shape
    if len(ori_dims) == 3:
        mu1 = mu1.reshape((-1, 2))
        mu2 = mu2.reshape((-1, 2))
        P1 = P1.reshape((-1, 2, 2))
        P2 = P2.reshape((-1, 2, 2))
    s = 1000     # Scale to be more stable
    S = P1 + P2
    S = S * s ** 2
    diff = mu1 - mu2
    diff = diff * s
    ret = diff[:, None, :] @ inv(S) @ diff[:, :, None]
    if len(ori_dims) == 3:
        ret = ret.reshape((ori_dims[0], ori_dims[1]))
    return ret


class Gaussians:
    def __init__(self, mu=None, covariances=None):
        self.mu = np.zeros((0, 2))  # means
        self.P = np.zeros((0, 2, 2))  # covariances
        if mu is not None:
            self.mu = np.asarray(mu)
            if self.mu.ndim == 1:
                self.mu = self.mu[None, :]
        if covariances is not None:
            self.P = np.asarray(covariances)
            if self.P.ndim == 2:
                self.P = self.P[None, :, :]
        assert self.mu.shape[1] == 2, self.mu.shape
        assert self.P.shape[1] == 2 and self.P.shape[2] == 2, self.P.shape

    def __copy__(self):
        return Gaussians(deepcopy(self.mu), deepcopy(self.P))

    def __add__(self, x):
        return Gaussians(vstack((self.mu, x.mu)), vstack((self.P, x.P)))

    def __getitem__(self, item):
        return Gaussians(self.mu[item:item+1], self.P[item:item+1])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.mu)


class PoissonPointProcess:
    """ Gaussian mixture density for a poisson point process """

    def all_probabilities(self, x):
        assert x.ndim == 2
        assert x.shape[1] == 2
        mu = np.repeat(self.gm.mu[None,], len(x), axis=0)
        P = np.repeat(self.gm.P[None,], len(x), axis=0)
        w = np.repeat(self.w[None,], len(x), axis=0)[:, :, 0]
        x = np.repeat(x[:, None, ], len(self), axis=1)
        res = gaussian_pdf(x, mu, P) * w
        res = np.sum(res, axis=1, keepdims=True)
        return res

    def __init__(self, w=None, gm: Gaussians = None):
        self.w = np.zeros((0, 1)) if w is None else np.asarray(w)
        self.gm = Gaussians() if gm is None else gm
        if self.w.ndim == 1:
            self.w = self.w[:, None]
        assert len(self.w) == len(self.gm)
        assert self.w.shape[1] == 1
        assert self.w.ndim == 2
        assert np.all(self.w >= 0), f"Weights is {self.w, w}"

    def __add__(self, x):
        return PoissonPointProcess(vstack((self.w, x.w)), self.gm + x.gm)

    def __len__(self):
        return len(self.w)

    def __copy__(self):
        return PoissonPointProcess(deepcopy(self.w), deepcopy(self.gm))

    def __getitem__(self, index):
        return self.w[index:index + 1], self.gm[index]

    def prune(self, threshold):
        valid = self.w[:, 0] > threshold
        self.w = self.w[valid]
        self.gm.mu = self.gm.mu[valid]
        self.gm.P = self.gm.P[valid]

    def keep_top_k(self, threshold):
        top_k = np.argsort(-self.w)[:threshold]
        self.w = self.w[top_k]
        self.gm.mu = self.gm.mu[top_k]
        self.gm.P = self.gm.P[top_k]


class BernoulliMixture:

    def __init__(
            self,
            log_weight,
            parent,
            identifier,
            labels: np.ndarray = None,
            r: np.ndarray = None,
            gm: Gaussians = None,
            parent_label: np.ndarray = None,
            age: np.ndarray = None,
            associated_id: np.ndarray = None,
            complete_track_lengths: list = None,
            incomplete_track_lengths: list = None,
    ):
        self.parent = parent
        self.identifier = identifier
        self.l = log_weight  # log_weight
        self.labels = np.zeros((0, 1))  # labels
        self.next_label = 1
        self.r = np.zeros((0, 1))  # probabilities_of_existence
        self.gm = Gaussians()
        self.age = np.zeros((0, 1))
        self.associated_id = np.zeros((0, 1))
        self.complete_track_lengths = [] if complete_track_lengths is None else complete_track_lengths
        self.incomplete_track_lengths = [] if incomplete_track_lengths is None else incomplete_track_lengths
        if labels is not None:
            self.labels = np.asarray(labels)
            if len(self.labels.shape) == 1:
                self.labels = self.labels[:, None]
            max_label = np.max(self.labels) if len(self.labels) > 0 else 0
            self.next_label = max_label + 1
        if r is not None:
            self.r = np.asarray(r)
            if len(self.r.shape) == 1:
                self.r = self.r[:, None]
            assert np.all(self.r >= 0), f"r is {self.r}"
        if age is not None:
            self.age = np.asarray(age)
            if len(self.age.shape) == 1:
                self.age = self.age[:, None]
        if gm is not None:
            self.gm = gm
        if parent_label is not None:
            self.parent_label = np.asarray(parent_label)
            if len(self.parent_label.shape) == 1:
                self.parent_label = self.parent_label[:, None]
        else:
            self.parent_label = -np.ones_like(self.labels)
        if associated_id is not None:
            self.associated_id = np.asarray(associated_id)
            if len(self.associated_id.shape) == 1:
                self.associated_id = self.associated_id[:, None]
        assert ~(self.labels == 0).any(), f"labels is {self.labels}"

    def __len__(self):
        return len(self.r)

    def __copy__(self):
        return BernoulliMixture(
            log_weight=self.l,
            parent=self.parent,
            identifier=self.identifier,
            labels=np.copy(self.labels),
            r=deepcopy(self.r),
            gm=deepcopy(self.gm),
            parent_label=np.copy(self.parent_label),
            associated_id=np.copy(self.associated_id),
            age=np.copy(self.age),
            complete_track_lengths=self.complete_track_lengths,
            incomplete_track_lengths=self.incomplete_track_lengths,
        )

    def __getitem__(self, i):
        return self.r[i:i + 1], self.labels[i:i + 1], self.gm[i]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __add__(self, other):
        if isinstance(other, BernoulliMixture):
            new_labels = np.arange(len(other))[:, None] + self.next_label
            r = other.r
            gm = other.gm
            age = other.age
            assoc_id = other.associated_id
            complete_track_lengths = self.complete_track_lengths + other.complete_track_lengths
            incomplete_track_lengths = self.incomplete_track_lengths + other.incomplete_track_lengths
        elif isinstance(other, PoissonPointProcess):
            new_labels = np.arange(len(other))[:, None] + self.next_label
            r = other.w
            gm = other.gm
            l = 0
            age = np.zeros_like(r)
            assoc_id = np.zeros_like(r)
            complete_track_lengths = self.complete_track_lengths
            incomplete_track_lengths = self.incomplete_track_lengths
        elif isinstance(other, tuple):
            new_labels = [[self.next_label]]
            r = other[0]
            gm = other[1]
            l = 0
            age = [[0]]
            assoc_id = [[0]]
            complete_track_lengths = self.complete_track_lengths
            incomplete_track_lengths = self.incomplete_track_lengths
        else:
            raise NotImplementedError
        return BernoulliMixture(
            log_weight=self.l+l,
            parent=self.parent,
            identifier=self.identifier,
            labels=concatenate((self.labels, new_labels), axis=0),
            r=concatenate((self.r, r), axis=0),
            gm=self.gm + gm,
            parent_label=concatenate(
                (self.parent_label, -np.ones_like(new_labels)), axis=0),
            associated_id=concatenate((self.associated_id, assoc_id), axis=0),
            age=concatenate((self.age, age), axis=0),
            complete_track_lengths=complete_track_lengths,
            incomplete_track_lengths=incomplete_track_lengths,
        )

    def append(self, other):
        if isinstance(other, BernoulliMixture):
            new_labels = np.arange(len(other)) + self.next_label
            self.next_label = self.next_label + len(other)
            self.labels = concatenate((self.labels, new_labels), axis=0)
            self.r = concatenate((self.r, other.r), axis=0)
            self.gm = self.gm + other.gm
            self.parent_label = concatenate(
                (self.parent_label, other.parent_label), axis=0)
            self.age = concatenate((self.age, other.age), axis=0)
            self.associated_id = concatenate(
                (self.associated_id, other.associated_id), axis=0)
            self.complete_track_lengths = self.complete_track_lengths + other.complete_track_lengths
            self.incomplete_track_lengths = self.incomplete_track_lengths + other.incomplete_track_lengths
        elif isinstance(other, PoissonPointProcess):
            new_labels = np.arange(len(other)) + self.next_label
            self.next_label = self.next_label + len(other)
            self.labels = concatenate((self.labels, new_labels), axis=0)
            self.r = concatenate((self.r, other.w), axis=0)
            self.gm = self.gm + other.gm
            self.parent_label = concatenate(
                (self.parent_label, -np.ones_like(new_labels)), axis=0)
            self.age = concatenate((self.age, np.zeros_like(other.w)), axis=0)
            self.associated_id = concatenate(
                (self.associated_id, np.zeros_like(other.w)), axis=0)
        elif isinstance(other, tuple):
            self.labels = concatenate((self.labels, [self.next_label]), axis=0)
            self.next_label = self.next_label + 1
            self.r = concatenate((self.r, other[0]), axis=0)
            self.gm = self.gm + other[1]
            self.parent_label = concatenate(
                (self.parent_label, [-1]), axis=0)
            self.age = concatenate((self.age, [0]), axis=0)
            self.associated_id = concatenate(
                (self.associated_id, [[0]]), axis=0)
        else:
            raise NotImplementedError

    def get_child(self, new_identifier):
        bm = BernoulliMixture(
            log_weight=self.l,
            parent=self.identifier,
            identifier=new_identifier,
            labels=np.copy(self.labels),
            r=np.copy(self.r),
            gm=self.gm.__copy__(),
            parent_label=np.copy(self.parent_label),
            associated_id=np.copy(self.associated_id),
            age=np.copy(self.age),
            complete_track_lengths=deepcopy(self.complete_track_lengths),
            incomplete_track_lengths=deepcopy(self.incomplete_track_lengths),
        )
        bm.next_label = self.next_label
        return bm

    def prune(self, threshold):
        valid = self.r[:, 0] > threshold
        self.r = self.r[valid]
        self.labels = self.labels[valid]
        self.gm.mu = self.gm.mu[valid]
        self.gm.P = self.gm.P[valid]
        self.parent_label = self.parent_label[valid]
        self.age = self.age[valid]
        self.associated_id = self.associated_id[valid]

    def remove(self, inds):
        valid = np.ones(len(self), dtype=bool)
        valid[inds] = False
        self.r = self.r[valid]
        self.labels = self.labels[valid]
        self.gm.mu = self.gm.mu[valid]
        self.gm.P = self.gm.P[valid]
        self.parent_label = self.parent_label[valid]
        self.age = self.age[valid]
        self.associated_id = self.associated_id[valid]


class MultiBernoulliMixture:
    def __init__(self, mixtures: list = None):
        self.bm = list()
        if mixtures is not None:
            self.bm = mixtures

    def __len__(self):
        return len(self.bm)

    def __getitem__(self, index):
        return self.bm[index]

    def __iter__(self):
        return iter(self.bm)

    def __add__(self, other):
        if isinstance(other, MultiBernoulliMixture):
            return MultiBernoulliMixture(
                self.bm + other.bm)
        elif isinstance(other, BernoulliMixture):
            return MultiBernoulliMixture(self.bm + [other])
        else:
            raise NotImplementedError

    def append(self, other):
        if isinstance(other, MultiBernoulliMixture):
            self.bm += other.bm
        elif isinstance(other, BernoulliMixture):
            self.bm.append(other)
        else:
            raise NotImplementedError

    def normalize(self):
        log_l = np.asarray([bm.l for bm in self.bm])
        # Make it more stable with small values
        log_l = log_l - np.max(log_l)
        p = exp(log_l)
        assert (p > 0.00001).any(), f"{np.max(log_l), p,  np.asarray([bm.l for bm in self.bm])}"
        p = maximum(p, 0.00001)
        assert not np.isnan(p).any(), f"p is nan: p {p}, log_l {log_l}"
        p = p / sum(p)
        new_log_likelihoods = log(p)
        for bm, l in zip(self.bm, new_log_likelihoods):
            bm.l = l
        # Sort by likelihood
        self.bm = sorted(self.bm, key=lambda x: x.l, reverse=True)

    def keep_top_k(self, threshold):
        log_l = np.asarray([float(bm.l) for bm in self.bm])
        top_k = np.argsort(-log_l)[:threshold]
        self.bm = \
            [self.bm[int(i)] for i in top_k]

    def top_likelihood(self):
        log_likelihoods = np.asarray([bm.l for bm in self.bm])
        top = exp(np.max(log_likelihoods))
        return top

    def get_best_hypothesis(self):
        log_likelihoods = np.asarray([bm.l for bm in self.bm])
        top = np.argmax(log_likelihoods)
        return self.bm[top]

    def prune_old(self, threshold):
        # NOTE: WARNING it could remove all hypotheses
        self.bm = [bm for bm in self.bm if np.exp(bm.l) > threshold]

    def prune(self, threshold):
        likelihoods = np.asarray([np.exp(bm.l)for bm in self.bm])
        likelihoods = likelihoods / likelihoods.sum()
        cumsum = likelihoods.cumsum()
        cumsum = np.concatenate(([0], cumsum))[:-1]
        self.bm = [bm for i, bm in enumerate(self.bm) if cumsum[i] < threshold]

    def get_total_probability(self):
        log_likelihoods = np.asarray([bm.l for bm in self.bm])
        return np.exp(log_likelihoods).sum()

