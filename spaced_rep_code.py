"""This file contains functions which are used to generate the log-likelihood
for different memory models and other code required to run the experiments in
the manuscript."""

import multiprocessing as MP
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

# Constants

TIME_SCALE = 60 * 60 * 24

MODEL_POWER = True
B = [1]
POWER_B = 1


def get_unique_user_lexeme(data_dict):
    """Get all unique (user, lexeme) pairs."""
    pairs = set()
    for u_id in data_dict.keys():
        pairs.update([(u_id, x) for x in data_dict[u_id].keys()])

    return sorted(pairs)


def max_unif(N, sum_D):
    """Find maximum value of N * log(x) - x * sum_D"""
    x_max = N / sum_D
    return N * np.log(x_max) - sum_D * x_max


def max_memorize(n_0, a, b, recalled, Ds,
                 q_fixed=None, n_max=np.inf, n_min=0, verbose=True):
    """Return max_{over q} memorizeLL."""

    # TODO: Currently, these are not true.
    # n_max=1440, n_min=1/36500000
    # maximum forgetting rate is clipped at 1 minute for exp(-1) forgetting and
    # minimum forgetting rate is that exp(-1) chance of forgetting after 100,000 years.

    assert len(recalled) == len(Ds), "recalled and t_is are not of the same length."

    N = len(Ds)
    n_t = n_0
    log_sum = 0
    int_sum = 0

    n_ts = []
    m_dts = []

    not_warned_min, not_warned_max = True, True

    n_correct, n_wrong = 0, 0

    with warnings.catch_warnings():
        warnings.simplefilter('once' if verbose else 'ignore')

        for D, recall in zip(Ds, recalled):
            if MODEL_POWER is False:
                m_dt = np.exp(-n_t * D)
            else:
                m_dt = (1 + POWER_B * D)**(-n_t)

            n_ts.append(n_t)
            m_dts.append(m_dt)

            if n_t < 1e-20:
                # log_sum += np.log(n_0) + n_correct * np.log(a) + n_wrong * np.log(b) + np.log(D)
                int_sum += n_t * (D ** 2) / 2
            else:
                if MODEL_POWER is False:
                    int_sum += D + np.expm1(-n_t * D) / n_t
                else:
                    int_sum += D - ((1 + POWER_B * D) ** (1 - n_t) - 1) / (POWER_B * (1 - n_t))

            if m_dt >= 1.0:
                log_sum = -np.inf
            else:
                log_sum += np.log1p(-m_dt)

            if recall:
                n_t *= (1 - a)
                n_correct += 1
            else:
                n_t *= (1 + b)
                n_wrong += 1

            n_t = min(n_max, max(n_min, n_t))

            if n_t == n_max and not_warned_max:
                if verbose:
                    warnings.warn('Max boundary hit.')
                not_warned_max = False

            if n_t == n_min and not_warned_min:
                if verbose:
                    warnings.warn('Min boundary hit.')
                not_warned_min = False

        if int_sum != 0:
            q_max = 1 / (4 * ((N / 2) / int_sum) ** 2) if q_fixed is None else q_fixed
        else:
            # If int_sum == 0, then LL should be -inf, not NaN
            q_max = 1.0

        LL = log_sum - (N / 2) * np.log(q_max) - (1 / q_max)**(0.5) * int_sum

    return {
        'q_max'            : q_max,
        'n_ts'             : n_ts,
        'm_dts'            : m_dts,
        'm_mean'           : np.mean(m_dts),
        'log_sum'          : log_sum,
        'int_sum'          : int_sum,
        'LL'               : LL,
        'max_boundary_hit' : not not_warned_max,
        'min_boundary_hit' : not not_warned_min,
        'n_max'            : n_max,
        'n_min'            : n_min
    }


def get_training_pairs(data_dict, pairs):
    """Returns the subset of pairs which have more than 3 reviews, i.e.
    the set for which we will be able to perform training using `n-1` reviews
    and testing for the last review."""

    training_pairs = []
    for u_id, l_id in pairs:
        if len(data_dict[u_id][l_id]) >= 3:
            training_pairs.append((u_id, l_id))

    return training_pairs


def calc_ll_arr(method, data_arr, alpha=None, beta=None,
                success_prob=0.49, eps=1e-10,
                all_mem_output=False, verbose=True):
    """Calculate LL for a given user_id, lexeme_id pair's data_arr."""
    n_0 = data_arr[0]['n_0']

    if method == 'uniform':
        sum_D = max(sum(x['delta_scaled'] for x in data_arr), eps)
        N     = len(data_arr)
        return max_unif(N, sum_D)
    elif method == 'memorize':
        recalled = np.asarray([x['p_recall'] > success_prob for x in data_arr])
        deltas   = np.asarray([x['delta_scaled'] for x in data_arr])
        deltas   = np.where(deltas <= 0, eps, deltas)
        op       = max_memorize(n_0, alpha, beta, recalled, deltas, verbose=verbose)
        if not all_mem_output:
            return op['LL']
        else:
            return op
    else:
        raise ValueError("Invalid method: {}".format(method))


def calc_user_LL_dict(data_dict, alpha, beta, lexeme_difficulty, map_lexeme,
                      success_prob=0.49, n_procs=None, pairs=None, verbose=True,
                      training=False):
    """Calculate LL while assuming that the LL factors are the same per user
    instead of setting them for each (user, lexeme) pair.

    If `training` is True, then the LL calculation is done only for the
    first n-1 entries instead of for all events in the sequence.
    """

    u_l_dict = defaultdict(lambda: defaultdict(lambda: {}))

    lexeme_difficulty = np.abs(lexeme_difficulty)

    global stat_worker
    def stat_worker(params):
        u_id = params
        data_per_lexeme = data_dict[u_id]

        n_0s = []
        Ns, sumDs = [], []
        log_sums, int_sums = [], []
        all_mem_op = []

        # The tests for all lexemes.
        all_tests = []

        lexeme_ids = sorted(data_per_lexeme.keys())
        valid_lexeme_ids = []

        for l_id in lexeme_ids:
            arr = data_per_lexeme[l_id]

            if training:
                if len(arr) < 3:
                    # Cannot calculate the LL for sequences shorter than 3
                    # elements if we are looking to train + test with these
                    # sequences.
                    continue
                else:
                    # Ignore the last review, which we will use for testing.
                    all_tests.append(arr[-1])
                    # Append the test before truncating arr
                    arr = arr[:-1]

            valid_lexeme_ids.append(l_id)
            n_0 = arr[0]['n_0']
            n_0s.append(n_0)
            Ns.append(len(arr))
            sumDs.append(sum(x['delta_scaled'] for x in arr))
            mem_res = calc_ll_arr('memorize', arr,
                                  alpha=alpha, beta=beta,
                                  success_prob=success_prob, all_mem_output=True,
                                  verbose=verbose)
            log_sums.append(mem_res['log_sum'])
            int_sums.append(mem_res['int_sum'])
            all_mem_op.append(mem_res)

        c_unif = np.sum(Ns) / np.sum(sumDs)
        q_max = 1 / (4 * ((np.sum(Ns) / 2) / np.sum(int_sums)) ** 2)
        res = {}
        for idx, l_id in enumerate(valid_lexeme_ids):
            res[l_id] = {
                'uniform_LL': Ns[idx] * np.log(c_unif) - sumDs[idx] * c_unif,
                'memorize_LL': log_sums[idx] + Ns[idx] * np.log(q_max) / 2 - (1 / q_max)**(0.5) * int_sums[idx],
                'mem_op': all_mem_op[idx],
                'q_max': q_max,
                'c_unif': c_unif
            }

            if training:
                res[l_id]['test'] = all_tests[idx]

        return u_id, res

    if n_procs is None:
        n_procs = MP.cpu_count()

    user_ids = sorted(set([u_id for u_id, _ in pairs]))

    with MP.Pool(n_procs) as pool:
        if n_procs > 1:
            map_func = pool.map
        else:
            # This aids debugging.
            map_func = map

        for u_id, res in map_func(stat_worker, user_ids):
            u_l_dict[u_id] = res

    return u_l_dict


def max_threshold(n_0, a, b, recalled, Ds, w, p,
                  alpha_fixed=None, n_max=np.inf, n_min=0, verbose=True):
    """Return max_{over a} threshold-LL, unless alpha_fixed is provided.
    In that case, the LL is calculated for the given alpha.

    Note (relationship of the symbols with those used in the paper):
     - p is m_{th} in the paper.
     - alpha (also alpha_max) is c in the paper
     - w is 1/\zeta in the paper.
    """

    assert len(recalled) == len(Ds), "recalled and t_is are not of the same length."

    N = len(Ds)
    n_t = n_0
    log_sum = 0
    int_sum = 0

    n_ts = []
    m_dts = []
    tau_dts = []
    not_warned_min, not_warned_max = True, True

    n_correct, n_wrong = 0, 0

    sum_third = 0
    sum_second = 0
    with warnings.catch_warnings():
        warnings.simplefilter('once' if verbose else 'ignore')

        for D, recall in zip(Ds, recalled):
            if MODEL_POWER is True:
                tau = (np.exp(-np.log(p) / n_t) - 1) / B[0]
            else:
                tau = -np.log(p) / n_t
            if n_t < 1e-20 and p != 1.0:
                raise Exception("P should be 1 when n_t is not finite")
                # When n_t is too small, p should also be 1.
            elif n_t < 1e-20 and p == 1.0:
                D_ = np.max([D, 0.0001])
            else:
                D_ = np.max([D - tau, 0.0001])
            sum_third += w * np.expm1(-D_ / w)
            sum_second += -D_ / w

            n_ts.append(n_t)
            m_dts.append(np.exp(-n_t * D))
            tau_dts.append(tau)

            if recall:
                n_t *= a
                n_correct += 1
            else:
                n_t *= b
                n_wrong += 1

            n_t = min(n_max, max(n_min, n_t))

            if n_t == n_max and not_warned_max:
                if verbose:
                    warnings.warn('Max boundary hit.')
                not_warned_max = False

            if n_t == n_min and not_warned_min:
                if verbose:
                    warnings.warn('Min boundary hit.')
                not_warned_min = False

        if alpha_fixed is None:
            alpha_max = -N / sum_third
        else:
            alpha_max = alpha_fixed

        LL = N * np.log(np.max([alpha_max, 0.0001])) + sum_second + alpha_max * sum_third
        if np.isfinite(LL).sum() == 0:
            raise Exception("LL is not finite")

    return {
        'alpha_max': alpha_max,
        'n_ts': n_ts,
        'm_dts': m_dts,
        'm_mean': np.mean(m_dts),
        'log_sum': log_sum,
        'int_sum': int_sum,
        'LL': LL,
        'max_boundary_hit': not not_warned_max,
        'min_boundary_hit': not not_warned_min,
        'n_max': n_max,
        'n_min': n_min,
        'p': p,
        'w': w,

        'sum_second': sum_second,  # sum_i -D_i / w
        'sum_third': sum_third,  # sum_i w * (exp(-D_i / w) - 1)
        'N': N
    }


def calc_ll_arr_thres(
        method, data_arr, alpha=None, beta=None,
        success_prob=0.49, eps=1e-10, w_range=None, p_range=None,
        verbose=True, all_thres_output=True, alpha_fixed=None):

    assert method == 'threshold', "This function only computes the max_threshold LL."

    n_0      = data_arr[0]['n_0']
    recalled = np.asarray([x['p_recall'] > success_prob for x in data_arr])
    deltas   = np.asarray([x['delta_scaled'] for x in data_arr])
    deltas   = np.where(deltas <= 0, eps, deltas)
    best_LL  = None

    if w_range is None:
        w_range = [0.01, 0.1, 1, 10, 100]

    n_is = [n_0]

    with warnings.catch_warnings():
        warnings.simplefilter('once' if verbose else 'ignore')
        for x in data_arr:
            if x['p_recall'] > success_prob:
                n_is.append(n_is[-1] * alpha)
            else:
                n_is.append(n_is[-1] * beta)

        # Remove the last n_t
        n_is = np.array(n_is[:-1])

        if p_range is None:
            # In most cases p_ == 1, the np.unique limits useless iterations.
            if (n_is < 1e-20).sum() > 0:
                p_range = [1.0]
            else:
                p_ = np.exp(-deltas * n_is).max()
                p_range = np.unique(np.linspace(p_, 1, 4))

    for w in w_range:
        for p in p_range:
            op = max_threshold(n_0, a=alpha, b=beta, recalled=recalled,
                               Ds=deltas, p=p, w=w, verbose=verbose,
                               alpha_fixed=alpha_fixed)
            if best_LL is None or best_LL['LL'] < op['LL']:
                best_LL = op

    if all_thres_output:
        return best_LL
    else:
        return best_LL['LL']


def calc_LL_dict_threshold(data_dict, alpha, beta, pairs,
                           lexeme_difficulty, map_lexeme, success_prob=0.49,
                           p_range=None, w_range=None,
                           n_procs=None, verbose=True):
    """Calculate the LL of the threshold model optimized for each (user, item)
    pair."""
    u_l_dict = defaultdict(lambda: {})

    lexeme_difficulty = np.abs(lexeme_difficulty)

    global _max_threshold_worker
    def _max_threshold_worker(params):
        u_id, l_id = params
        arr = data_dict[u_id][l_id]
        op = calc_ll_arr_thres('threshold', arr, alpha=alpha, beta=beta,
                               success_prob=success_prob, all_thres_output=True,
                               verbose=verbose)
        return u_id, l_id, {'threshold_op': op, 'threshold_LL': op['LL']}

    if n_procs is None:
        n_procs = MP.cpu_count()

    with MP.Pool() as pool:
        for u_id, l_id, res in pool.map(_max_threshold_worker, pairs):
            u_l_dict[u_id][l_id] = res

    return u_l_dict


def calc_user_ll_arr_thres(
        method, user_data_dict, alpha=None, beta=None,
        success_prob=0.49, eps=1e-10, w_range_init=None, p_range_init=None,
        training=False, verbose=True, all_thres_output=True):
    """Calculates the best-LL for a given user, by computing it across all
    items the user has touched.

    If `training` is True, then only consider the first 'n - 1' reviews
    of the user/lexme pairs, ignoring sequences smaller than 2.
    """

    assert method == 'threshold', "This function only computes the max_threshold LL."

    total_sum_second = defaultdict(lambda: 0)
    total_sum_third = defaultdict(lambda: 0)
    total_N = 0
    p_ = 0.0

    if w_range_init is None:
        w_range = [0.01, 0.1, 1, 10, 100]
    else:
        w_range = w_range_init

    if p_range_init is None:
        for l_id in user_data_dict.keys():
            data_arr = user_data_dict[l_id]

            n_0      = data_arr[0]['n_0']
            deltas   = np.asarray([x['delta_scaled'] for x in data_arr])
            deltas   = np.where(deltas <= 0, eps, deltas)

            n_is = [n_0]

            with warnings.catch_warnings():
                warnings.simplefilter('once' if verbose else 'ignore')
                for x in data_arr:
                    if x['p_recall'] > success_prob:
                        n_is.append(n_is[-1] * alpha)
                    else:
                        n_is.append(n_is[-1] * beta)

                # Remove the last n_t
                n_is = np.array(n_is[:-1])

                # In most cases p_ == 1, the np.unique limits useless iterations.
                if (n_is < 1e-20).sum() > 0:
                    p_ = 1.0
                else:
                    p_ = max(p_, np.exp(-deltas * n_is).max())

        if p_ < 1.0:
            p_range = np.linspace(p_, 1, 4)
        else:
            # if p_ == 1.0, then no point taking linspace.
            p_range = [p_]
    else:
        p_range = p_range_init

    for l_id in user_data_dict.keys():
        data_arr = user_data_dict[l_id]

        if training:
            if len(data_arr) < 3:
                # Cannot calculate the LL for training and have a test unless
                # there are at least 3 reviews.
                continue
            else:
                # Calculate the LL only using the first 'n-1' reviews.
                data_arr = data_arr[:-1]

        total_N += len(data_arr)

        n_0      = data_arr[0]['n_0']
        recalled = np.asarray([x['p_recall'] > success_prob for x in data_arr])
        deltas   = np.asarray([x['delta_scaled'] for x in data_arr])
        deltas   = np.where(deltas <= 0, eps, deltas)

        for w in w_range:
            for p in p_range:
                op = max_threshold(n_0, a=alpha, b=beta, recalled=recalled,
                                   Ds=deltas, p=p, w=w, verbose=verbose)
                total_sum_second[w, p] += op['sum_second']
                total_sum_third[w, p] += op['sum_third']

    best_LL  = None
    for w, p in total_sum_second.keys():
        alpha_max_user = - total_sum_third[w, p] / total_N
        LL = total_N * alpha_max_user + total_sum_second[w, p] + alpha_max_user * total_sum_third[w, p]
        if best_LL is None or best_LL['LL'] < LL:
            best_LL = {
                'LL': LL,
                'w': w,
                'p': p,
                'sum_third': total_sum_third[w, p],
                'sum_second': total_sum_second[w, p],
                'alpha_max_user': alpha_max_user
            }

    if all_thres_output:
        return best_LL
    else:
        return best_LL['LL']


def calc_user_LL_dict_threshold(data_dict, alpha, beta, pairs,
                                lexeme_difficulty, map_lexeme, success_prob=0.49,
                                p_range=None, w_range=None, training=False,
                                n_procs=None, verbose=True):
    """Calculate the LL of the threshold model optimized for each user.

    if `training` is True, then it computes the likelihood only for the first
    `n - 1` entries instead of for all 'n' reviews.
    """

    u_l_dict = defaultdict(lambda: {})

    lexeme_difficulty = np.abs(lexeme_difficulty)

    if n_procs is None:
        n_procs = MP.cpu_count()

    global _max_user_c_worker
    def _max_user_c_worker(params):
        u_id = params
        best_LL = calc_user_ll_arr_thres('threshold',
                                         user_data_dict=data_dict[u_id],
                                         alpha=alpha, beta=beta,
                                         success_prob=success_prob,
                                         training=training,
                                         all_thres_output=True,
                                         verbose=verbose)
        return u_id, best_LL

    with MP.Pool() as pool:
        u_best_alpha = dict(pool.map(_max_user_c_worker, data_dict.keys()))

    global _max_user_threshold_worker
    def _max_user_threshold_worker(params):
        u_id, l_id = params
        alpha_max_user = u_best_alpha[u_id]['alpha_max_user']

        w_range = [u_best_alpha[u_id]['w']]
        p_range = [u_best_alpha[u_id]['p']]

        arr = data_dict[u_id][l_id]
        if training:
            assert len(arr) >= 3, "Are you using `training_pairs` instead of" \
                                  " all pairs in the call?"
            test = arr[-1]
            # Append the test before truncating arr
            arr = arr[:-1]

        op = calc_ll_arr_thres('threshold', arr, alpha=alpha, beta=beta,
                               success_prob=success_prob,
                               all_thres_output=True, verbose=verbose,
                               alpha_fixed=alpha_max_user, w_range=w_range,
                               p_range=p_range)
        res = {'threshold_op': op, 'threshold_LL': op['LL']}

        if training:
            res['test'] = test

        return u_id, l_id, res

    with MP.Pool() as pool:
        for u_id, l_id, res in pool.map(_max_user_threshold_worker, pairs):
            u_l_dict[u_id][l_id] = res

    return u_l_dict


def merge_with_thres_LL(other_LL, thres_LL, pairs):
    """Merge the dictionaries containing the threshold-LL and other thresholds.
    Other_LL will be modified in place.
    """

    for u_id, l_id in pairs:
        for key in thres_LL[u_id][l_id]:
            other_LL[u_id][l_id][key] = thres_LL[u_id][l_id][key]

    return None


def get_all_durations(data_dict, pairs):
    """Generates all durations from the LL_dict or the data_dict."""

    def _get_duration(user_id, item_id):
        """Generates test/train/total duration for the given user_id, item_id pair."""
        session = data_dict[user_id][item_id]
        session_length = len(session)

        if session_length > 2:
            train_duration = session[-2]['timestamp'] - session[0]['timestamp']
            test_duration = session[-1]['timestamp'] - session[-2]['timestamp']
        else:
            train_duration = None
            test_duration = None

        if session_length > 1:
            total_duration = session[-1]['timestamp'] - session[0]['timestamp']
        else:
            total_duration = None

        return {
            'train_duration': train_duration,
            'test_duration': test_duration,
            'total_duration': total_duration,
            'session_length': session_length,
        }

    dur_dict = defaultdict(lambda: {})
    for u_id, i_id in pairs:
        dur_dict[u_id][i_id] = _get_duration(u_id, i_id)

    return dur_dict


def filter_by_duration(durations_dict, pairs, T, alpha, verbose=False):
    """Filter the (u_id, l_id) by selecting those which have the duration in
       [(1 - alpha) * T, (1 + alpha) * T]."""
    filtered_pairs = []

    for u_id, l_id in pairs:
        train_duration = durations_dict[u_id][l_id]['train_duration'] / TIME_SCALE
        if (1 - alpha) * T <= train_duration <= (1 + alpha) * T:
            filtered_pairs.append((u_id, l_id))

    count = len(filtered_pairs)

    total = len(pairs)
    if verbose:
        print('{} / {} = {:.2f}% sequences selected.'
              .format(count, total, count / total * 100.))

    return filtered_pairs


def filter_by_users(pairs, users_, verbose=False):
    """Filter the (u_id, l_id) by selecting those which have u_id \in users_."""
    filtered_pairs = []

    for u_id, l_id in pairs:
        if u_id in users_:
            filtered_pairs.append((u_id, l_id))

    count = len(filtered_pairs)

    total = len(pairs)
    if verbose:
        print('{} / {} = {:.2f}% sequences selected.'
              .format(count, total, count / total * 100.))

    return filtered_pairs


def calc_empirical_forgetting_rate(data_dict, pairs, return_base=False, no_norm=False):
    u_l_dict = defaultdict(lambda: defaultdict(lambda: None))

    base = {}
    base_count = {}
    for u_id, l_id in pairs:
        first_session = data_dict[u_id][l_id][0]
        res = (- np.log(max(0.01, min(0.99, first_session['p_recall'] + 1e-10))) / (first_session['delta_scaled'] + 0.1))
        if l_id not in base:
            base[l_id] = res
            base_count[l_id] = 1
        else:
            base[l_id]+=res
            base_count[l_id] += 1
    if return_base:
        return dict([(l_id, base[l_id] / base_count[l_id]) for l_id in base.keys()])
    for u_id, l_id in pairs:
        last_session = data_dict[u_id][l_id][-1]
        u_l_dict[u_id][l_id] = - np.log(max(0.01, min(0.99, last_session['p_recall'] + 1e-10))) / (last_session['delta_scaled'] + 0.1)
        if not no_norm:
            u_l_dict[u_id][l_id] = (u_l_dict[u_id][l_id]) / (base[l_id] / base_count[l_id])
        else:
            u_l_dict[u_id][l_id] = u_l_dict[u_id][l_id]

    return u_l_dict


def calc_top_k_perf(LL_dict, perf, pairs, quantile=0.25, min_reps=0,
                    max_reps=None, with_overall=False, with_threshold=False,
                    only_finite=True, with_uniform=True, whis=1.5):
    """Calculates the average and median performance of people in the
    top quantile by log-likelihood of following either strategy."""

    def check_u_l(u_id, l_id):
        return (not only_finite or np.isfinite(perf[u_id][l_id])) and \
            (min_reps <= 1 or len(LL_dict[u_id][l_id]['mem_op']['n_ts']) >= min_reps) and \
            (max_reps is None or len(LL_dict[u_id][l_id]['mem_op']['n_ts']) < max_reps)

    # global _get_top_k
    def _get_top_k(key):
        sorted_by_ll = sorted((LL_dict[u_id][l_id][key], u_id, l_id)
                              for u_id, l_id in pairs
                              if check_u_l(u_id, l_id))
        # print("counts {}".format(len(sorted_by_ll)))
        # Taking the quantile after limiting to only valid pairs.
        K = int(quantile * len(sorted_by_ll))
        # print("K: {}".format(K), quantile, len(sorted_by_ll[-K:]), len(sorted_by_ll[-K:]))
        return sorted_by_ll[-K:], sorted_by_ll[:K]

    top_memorize_LL, bottom_memorize_LL = _get_top_k('memorize_LL')

    top_common_u_l = set()
    bottom_common_u_l = set()
    if with_uniform:
        top_uniform_LL, bottom_uniform_LL = _get_top_k('uniform_LL')
        top_common_u_l = set([(u_id, l_id) for _, u_id, l_id in top_uniform_LL]).intersection(
                             [(u_id, l_id) for _, u_id, l_id in top_memorize_LL])

        bottom_common_u_l = set([(u_id, l_id) for _, u_id, l_id in bottom_uniform_LL]).intersection(
                                [(u_id, l_id) for _, u_id, l_id in bottom_memorize_LL])

    if with_threshold:
        top_threshold_LL, bottom_threshold_LL = _get_top_k('threshold_LL')

        # If we have already calculated a common set, then calculate
        # intersection with that set. Otherwise, take the intersection with
        # memorize directly.
        if not with_uniform:
            top_common_u_l = set([(u_id, l_id) for _, u_id, l_id in top_threshold_LL]).intersection(
                                 [(u_id, l_id) for _, u_id, l_id in top_memorize_LL])

            bottom_common_u_l = set([(u_id, l_id) for _, u_id, l_id in bottom_threshold_LL]).intersection(
                                    [(u_id, l_id) for _, u_id, l_id in bottom_memorize_LL])
        else:
            top_common_u_l = top_common_u_l.intersection(
                             [(u_id, l_id) for _, u_id, l_id in top_threshold_LL])
            bottom_common_u_l = bottom_common_u_l.intersection(
                                [(u_id, l_id) for _, u_id, l_id in bottom_threshold_LL])

    # global _perf_top_k
    def _perf_top_k(top_ll):
        return [perf[u_id][l_id]
                for _, u_id, l_id in top_ll
                if (u_id, l_id) not in top_common_u_l]

    def _perf_top_k_elem(top_ll):
        return [(u_id, l_id)
                for _, u_id, l_id in top_ll
                if (u_id, l_id) not in top_common_u_l]

    def _perf_bottom_k(bottom_ll):
        return [perf[u_id][l_id]
                for _, u_id, l_id in bottom_ll
                if (u_id, l_id) not in bottom_common_u_l]

    # Selecting only non-unique (u_id, l_id) from the top 25% of both.
    # Because several times, the same user, lexeme pair have likelihood in the
    # top 25%.
    # print("common {}".format(len(top_common_u_l)))
    perf_top_mem = _perf_top_k(top_memorize_LL)
    perf_top_mem_elem = _perf_top_k_elem(top_memorize_LL)
    perf_bottom_mem = _perf_bottom_k(bottom_memorize_LL)

    perc = [0.1, 0.2, 0.25, 0.30, 0.40, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]

    res = {
        'mem_top': pd.Series(perf_top_mem).describe(percentiles=perc),
        'mem_top_elem': perf_top_mem_elem,
        'mem_bottom': pd.Series(perf_bottom_mem).describe(percentiles=perc),
        'top_memorize_LL': top_memorize_LL,
        'bottom_memorize_LL': bottom_memorize_LL,

        'perf_top_mem': perf_top_mem,

        'top_common_u_l': top_common_u_l,
        'bottom_common_u_l': bottom_common_u_l
    }

    mem_min_whis = res['mem_top']['25%'] - (res['mem_top']['75%'] - res['mem_top']['25%']) * whis
    ind = np.asarray(perf_top_mem) < mem_min_whis
    if ind.sum() == 0:
        res['mem_top'].loc['min_whis'] = np.asarray(perf_top_mem).min()
    else:
        res['mem_top'].loc['min_whis'] = np.asarray(perf_top_mem)[ind].max()

    mem_max_whis = res['mem_top']['75%'] + (res['mem_top']['75%'] - res['mem_top']['25%']) * whis
    ind = np.asarray(perf_top_mem) > mem_max_whis
    if ind.sum() == 0:
        res['mem_top'].loc['max_whis'] = np.asarray(perf_top_mem).max()
    else:
        res['mem_top'].loc['max_whis'] = np.asarray(perf_top_mem)[ind].min()

    if with_uniform:
        perf_top_unif = _perf_top_k(top_uniform_LL)
        perf_top_unif_elem = _perf_top_k_elem(top_uniform_LL)
        perf_bottom_unif = _perf_bottom_k(bottom_uniform_LL)

        res.update({
            'unif_top': pd.Series(perf_top_unif).describe(percentiles=perc),
            'unif_top_elem': perf_top_unif_elem,
            'unif_bottom': pd.Series(perf_bottom_unif).describe(percentiles=perc),
            'top_uniform_LL': top_uniform_LL,
            'bottom_uniform_LL': bottom_uniform_LL,
            'perf_top_unif': perf_top_unif,
        })

        uni_min_whis = res['unif_top']['25%'] - (res['unif_top']['75%'] - res['unif_top']['25%']) * whis
        ind = np.asarray(perf_top_unif) < uni_min_whis
        if ind.sum() == 0:
            res['unif_top'].loc['min_whis'] = np.asarray(perf_top_mem).min()
        else:
            res['unif_top'].loc['min_whis'] = np.asarray(perf_top_mem)[ind].max()

        uni_max_whis = res['unif_top']['75%'] + (res['unif_top']['75%'] - res['unif_top']['25%']) * whis
        ind = np.asarray(perf_top_unif) > uni_max_whis
        if ind.sum() ==0:
            res['unif_top'].loc['max_whis'] = np.asarray(perf_top_unif).max()
        else:
            res['unif_top'].loc['max_whis'] = np.asarray(perf_top_unif)[ind].min()

    if with_threshold:
        perf_top_threshold = _perf_top_k(top_threshold_LL)
        perf_top_threshold_elem = _perf_top_k_elem(top_threshold_LL)
        perf_bottom_threshold = _perf_bottom_k(bottom_threshold_LL)

        res.update({
            'threshold_top': pd.Series(perf_top_threshold).describe(percentiles=perc),
            'threshold_top_elem': perf_top_threshold_elem,
            'threshold_bottom': pd.Series(perf_bottom_threshold).describe(percentiles=perc),
            'top_threshold_LL': top_threshold_LL,
            'bottom_threshold_LL': bottom_threshold_LL,
            'perf_top_threshold': perf_top_threshold,
        })

        thr_min_whis = res['threshold_top']['25%'] - (res['threshold_top']['75%'] - res['threshold_top']['25%']) * whis
        ind = np.asarray(perf_top_threshold)<thr_min_whis
        if ind.sum() ==0:
           res['threshold_top'].loc['min_whis'] = np.asarray(perf_top_threshold).min()
        else:
           res['threshold_top'].loc['min_whis'] = np.asarray(perf_top_threshold)[ind].max()

        thr_max_whis = res['threshold_top']['75%'] + (res['threshold_top']['75%'] - res['threshold_top']['25%']) * whis
        ind = np.asarray(perf_top_threshold) > thr_max_whis
        if ind.sum() == 0:
            res['threshold_top'].loc['max_whis'] = np.asarray(perf_top_threshold).max()
        else:
            res['threshold_top'].loc['max_whis'] = np.asarray(perf_top_threshold)[ind].min()
    if with_overall:
        res['unif_top_overall'] = pd.Series(perf[u_id][l_id] for _, u_id, l_id in top_uniform_LL).describe()
        res['mem_top_overall'] = pd.Series(perf[u_id][l_id] for _, u_id, l_id in top_memorize_LL).describe()

        res['unif_bottom_overall'] = pd.Series(perf[u_id][l_id] for _, u_id, l_id in bottom_uniform_LL).describe()
        res['mem_bottom_overall'] = pd.Series(perf[u_id][l_id] for _, u_id, l_id in bottom_memorize_LL).describe()

        if with_threshold:
            res['threshold_top_overall'] = pd.Series(perf[u_id][l_id] for _, u_id, l_id in top_threshold_LL).describe()
            res['threshold_bottom_overall'] = pd.Series(perf[u_id][l_id] for _, u_id, l_id in bottom_threshold_LL).describe()

    return res


def calc_top_memorize(LL_dict, perf, pairs, min_reps=0,
                      max_reps=None, with_overall=False, with_threshold=False,
                      only_finite=True, with_uniform=True):
    """Calculates the average and median performance of people in the
    top quantile by log-likelihood of following either strategy."""
    from scipy.stats.stats import pearsonr

    def check_u_l(u_id, l_id):
        return (not only_finite or np.isfinite(perf[u_id][l_id])) and \
            (min_reps <= 1 or len(LL_dict[u_id][l_id]['mem_op']['n_ts']) >= min_reps) and \
            (max_reps is None or len(LL_dict[u_id][l_id]['mem_op']['n_ts']) < max_reps)

    quantile = 0.1

    def _get_top_k(key):
        sorted_by_ll = sorted((LL_dict[u_id][l_id][key], u_id, l_id)
                              for u_id, l_id in pairs
                              if check_u_l(u_id, l_id))
        K = int(quantile * len(sorted_by_ll))
        restricted = sorted_by_ll[:K] + sorted_by_ll[-K:]
        coeff = pearsonr([ll for ll, u_id, l_id in restricted], [perf[u_id][l_id] for _, u_id, l_id in restricted])
        if ~np.isfinite(coeff[0]):
            # print("NAN", restricted)
            pass
        return coeff

    top_memorize_LL = _get_top_k('memorize_LL')
    top_uniform_LL = _get_top_k('uniform_LL')
    top_threshold_LL = _get_top_k('threshold_LL')
    return top_memorize_LL, top_uniform_LL, top_threshold_LL


def plot_perf_by_reps_boxed(reps_perf_pairs, median=True,
                            with_threshold=False, std=False,
                            max_rev=5, stats=None):
    """Produces plots comparing different scheduling techniques."""

    def mean_var(data, color, positions, hatch):
        labels = []
        means = []
        err = []
        for d, pos in zip(data, positions):
            confidence = 0.95
            m, se = np.mean(d), stats.sem(d)
            n = len(d)
            h = se * stats.t._ppf((1 + confidence) / 2., n - 1)
            means.append(m)
            err.append(h)
            labels.append(pos)
        plt.errorbar(labels, means, yerr=err, fmt='o', capsize=5, elinewidth=3, capthick=2.5, color=color)

    def boxify(data, color, positions, hatch, w):
        boxprops = dict(linewidth=0, color=None, facecolor=color, alpha=0.6)
        whiskerprops = dict(linewidth=3, alpha=0.0)
        medianprops = dict(linewidth=0.9, color='black')
        meanlineprops = dict(linestyle='--', markersize=12.5, color=color)
        capprops = dict(linestyle='-', linewidth=3, color='black', alpha=0)
        bp = plt.boxplot(data, notch=False, showmeans=False, positions=positions,
                         boxprops=boxprops, meanprops=meanlineprops,
                         whiskerprops=whiskerprops, medianprops=medianprops,
                         capprops=capprops, patch_artist=True,
                         widths=[w] * len(data))  # a simple case with just one variable to boxplot
        for median in bp['medians']:
            x, y = median.get_data()
            dx = x[1] - x[0]
            median.set_data([x[0] + dx / 4, x[1] - dx / 4], y)

    def label_diff(stats, X, Y_1, Y_2, W, repetitions, ind):
        for i in range(len(repetitions)):
            p_val = stats[int(repetitions[i])][ind][1]
            y = 1.1 * max(Y_1[i], Y_2[i])

            props = {
                'connectionstyle': 'bar',
                'arrowstyle': '-',
                'shrinkA': 0,
                'shrinkB': 0,
                'linewidth': 0.7
            }

            if p_val < 0.05:
                plt.gca().annotate("*", xy=(X[i], y + 0.01), zorder=10, fontsize=10)
            plt.gca().annotate('', xy=(X[i], y), xytext=(X[i] + 3 * W / 4, y), arrowprops=props)

    plt.figure()
    c1, c2, c3 = sns.color_palette("Set2", n_colors=3)

    mem_lows, mem_highs, mem_means, mem_meds, mem_stds, mem_counts, mem_whis_lows, mem_whis_highs, mem_samples = [], [], [], [], [], [], [], [], []
    unif_lows, unif_highs, unif_means, unif_meds, unif_stds, unif_counts, unif_whis_lows, unif_whis_highs, unif_samples = [], [], [], [], [], [], [], [], []
    thres_lows, thres_highs, thres_means, thres_meds, thres_stds, thres_counts, thres_whis_lows, thres_whis_highs, thres_samples = [], [], [], [], [], [], [], [], []

    for reps, top_k in reps_perf_pairs:
        mem_low, mem_high, mem_mean, mem_med, mem_std, mem_count, mem_w_max, mem_w_min = [top_k['mem_top'][x] for x in ['25%', '75%', 'mean', '50%', 'std', 'count', 'max_whis', 'min_whis']]
        mem_lows.append(mem_low)
        mem_highs.append(mem_high)
        mem_means.append(mem_mean)
        mem_meds.append(mem_med)
        mem_stds.append(mem_std)
        mem_counts.append(mem_count)
        mem_whis_lows.append(mem_w_min)
        mem_whis_highs.append(mem_w_max)
        mem_samples.append(top_k['perf_top_mem'])

        unif_low, unif_high, unif_mean, unif_med, unif_std, unif_count, unif_w_max, unif_w_min = [top_k['unif_top'][x] for x in ['25%', '75%', 'mean', '50%', 'std', 'count', 'max_whis', 'min_whis']]
        unif_lows.append(unif_low)
        unif_highs.append(unif_high)
        unif_means.append(unif_mean)
        unif_meds.append(unif_med)
        unif_stds.append(unif_std)
        unif_counts.append(unif_count)
        unif_whis_lows.append(unif_w_min)
        unif_whis_highs.append(unif_w_max)
        unif_samples.append(top_k['perf_top_unif'])

        if with_threshold:
            thres_low, thres_high, thres_mean, thres_med, thres_std, thres_count, thres_w_max, thres_w_min = [top_k['threshold_top'][x] for x in ['25%', '75%', 'mean', '50%', 'std', 'count', 'max_whis', 'min_whis']]
            thres_lows.append(thres_low)
            thres_highs.append(thres_high)
            thres_means.append(thres_mean)
            thres_meds.append(thres_med)
            thres_stds.append(thres_std)
            thres_counts.append(thres_count)
            thres_whis_lows.append(unif_w_min)
            thres_whis_highs.append(unif_w_max)
            thres_samples.append(top_k['perf_top_threshold'])

    repetitions = np.array([reps for reps, _ in reps_perf_pairs])
    yerr = np.array(mem_stds) / np.sqrt(np.array(mem_counts)) if std else [(x - y) * 1.57 / np.sqrt(c) for (x, y, c) in zip(mem_highs, mem_lows, mem_counts)]
    yerr = np.array(mem_stds) / np.sqrt(np.array(mem_counts)) if std else [(x - y) * 1.57 / np.sqrt(c) for (x, y, c) in zip(mem_highs, mem_lows, mem_counts)]

    W = 4
    w = 0.8
    # print(len(mem_samples), len(mem_samples[0]))
    if median is False:
        mean_var(mem_samples, color=c1, hatch=r"", positions=(repetitions) * W - w)
        mean_var(thres_samples, color=c2, hatch=r"", positions=(repetitions) * W)
        mean_var(unif_samples, color=c3, hatch=r"", positions=(repetitions) * W + w)
    else:
        boxify(thres_samples, color=c2, hatch=r"", positions=(repetitions) * W - w, w=0.75 * w)
        boxify(mem_samples, color=c1, hatch=r"", positions=(repetitions) * W, w=0.75 * w)
        boxify(unif_samples, color=c3, hatch=r"", positions=(repetitions) * W + w, w=0.75 * w)

    label_diff(stats, repetitions * W - w, thres_highs, mem_highs, w, repetitions, ind=0)
    label_diff(stats, repetitions * W, unif_highs, mem_highs, w, repetitions, ind=1)
    plt.xticks(repetitions * W, repetitions)

    hMem = mpatches.Patch(facecolor=c1, linewidth=6, hatch=r"", label='Memorize')
    hThre = mpatches.Patch(facecolor=c2, linewidth=6, hatch=r'', label='Threshold')
    hUni = mpatches.Patch(facecolor=c3, linewidth=6, hatch=r'', label='Uniform')
    plt.xlim(repetitions.min() * W - w * 1.5, max_rev * W + w * 1.5)

    if with_threshold:
        plt.legend((hThre, hMem, hUni), ('Threshold', '\\textsc{Memorize}', 'Uniform'), loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2), handletextpad=0.1, columnspacing=0.5)
    else:
        plt.legend((hThre, hMem, hUni), ('Threshold', '\\textsc{Memorize}', 'Uniform'), loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2))

