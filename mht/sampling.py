import numpy as np
from scipy.optimize import linear_sum_assignment

np.set_printoptions(precision=2, suppress=True, edgeitems=20, linewidth=1000)


def murty(
        costs: np.array,
        max_hypotheses: int = None,
        min_increment: float = None,
        invalid_cost=np.inf,
):
    if costs.size == 0:
        return [(np.zeros(0), np.zeros(0))], [1]
    hypotheses = list()
    likelihoods = list()
    partitions = None
    J, I = costs.shape
    I = int((I - J) / 2)

    while True:
        ret = murty_stage(
            costs=costs,
            invalid_cost=invalid_cost,
            partitions=partitions,
        )
        if ret is None:
            break
        partitions, (rows, cols) = ret
        likelihood = np.exp(-np.sum(costs[rows, cols]))
        hypotheses.append((rows, cols))
        likelihoods.append(likelihood)
        if max_hypotheses is not None and len(hypotheses) >= max_hypotheses:
            break
        if min_increment is not None:
            if likelihoods[-1] / likelihoods[0] < min_increment:
                break
    return hypotheses, likelihoods


def murty_stage(
        costs: np.array,
        invalid_cost=np.inf,
        partitions: tuple = None,
):
    """
    Sampler made for a cost matrix that is
        [n, m+n]
    in dimension and has only positive values that are log likelihoods.

    :param costs:
    :param invalid_cost:
    :return:
    """
    def partition(rows, cols, costs, p_in=None, p_out=None):
        J, I = costs.shape  # Number of Detections and Objects
        I = int((I - J) / 2)
        partitions_in, partitions_out, total_costs = list(), list(), list()
        if p_in is None and p_out is None:
            p_in = (np.zeros(0, dtype=int), np.zeros(0, dtype=int))
            p_out = (np.zeros(0, dtype=int), np.zeros(0, dtype=int))

        _rows, _cols = list(), list()
        for i, (r, c) in enumerate(zip(rows, cols)):
            if r in p_in[0]:
                assert c in p_in[1]
            else:
                _rows.append(r)
                _cols.append(c)
        rows, cols = np.asarray(_rows), np.asarray(_cols)

        for i, (r, c) in enumerate(zip(rows, cols)):
            _p_in = (
                np.concatenate([p_in[0], rows[:i]]),
                np.concatenate([p_in[1], cols[:i]])
            )
            _p_out = (
                np.concatenate([p_out[0], rows[i:i+1]]),
                np.concatenate([p_out[1], cols[i:i+1]])
            )
            # Add split to list
            if c < I:
                _p_out = (
                    np.concatenate([_p_out[0], rows[i:i+1]]),
                    np.concatenate([_p_out[1], cols[i:i+1]+I])
                )
            elif c < 2 * I:
                _p_out = (
                    np.concatenate([_p_out[0], rows[i:i+1]]),
                    np.concatenate([_p_out[1], cols[i:i+1]-I])
                )

            partitions_in.append(_p_in)
            partitions_out.append(_p_out)
            _costs = np.copy(costs)
            _costs[_p_out[0], _p_out[1]] = invalid_cost
            _costs[_p_in[0], :] = invalid_cost
            _costs[:, _p_in[1]] = invalid_cost
            _costs[_p_in[0], _p_in[1]] = costs[_p_in[0], _p_in[1]]
            try:
                # Pre-check if potentially solvable
                _min_cost = np.min(_costs, axis=1)
                if np.any(_min_cost == invalid_cost):
                    total_costs.append(invalid_cost)
                    continue
                # Get optimal assignment to calculate total cost
                _rows, _cols = linear_sum_assignment(_costs)
                inds = np.argsort(_rows)
                _rows, _cols = _rows[inds], _cols[inds]
                total_costs.append(_costs[_rows, _cols].sum())
            except ValueError:
                total_costs.append(invalid_cost)
        return partitions_in, partitions_out, total_costs

    if partitions is None:
        # Find optimal assignment
        try:
            rows, cols = linear_sum_assignment(costs)
        except ValueError:
            return None
        inds = np.argsort(rows)
        rows, cols = rows[inds], cols[inds]
        partitions = partition(rows, cols, costs)
    else:
        # Find next suboptimal assignment
        h = np.argmin(partitions[2])
        partitions_in, partitions_out, total_costs = partitions
        p_in, p_out, _ = \
            partitions_in.pop(h), partitions_out.pop(h), total_costs.pop(h)
        _costs = np.copy(costs)
        _costs[p_in[0], :] = invalid_cost
        _costs[:, p_in[1]] = invalid_cost
        _costs[p_out[0], p_out[1]] = invalid_cost
        _costs[p_in[0], p_in[1]] = costs[p_in[0], p_in[1]]
        try:
            rows, cols = linear_sum_assignment(_costs)
        except ValueError:
            return None
        inds = np.argsort(rows)
        rows, cols = rows[inds], cols[inds]
        new_partitions = partition(rows, cols, costs, p_in=p_in, p_out=p_out)

        partitions = (
            partitions_in + new_partitions[0],
            partitions_out + new_partitions[1],
            total_costs + new_partitions[2]
        )

    # Calculate loss for each row

    return partitions, (rows, cols)

