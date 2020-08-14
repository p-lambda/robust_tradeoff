from pathlib import Path
from collections import defaultdict
from typing import NamedTuple
import pickle
import argparse
import warnings
from functools import reduce

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cvxpy as cp

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import scipy


class Result(NamedTuple):
    ''' Result struct that emulates scipy.minimize return result'''
    x: np.array


def sparse_diff(array, n=1, axis=-1):
    """
    A ported sparse version of np.diff.
    Uses recursion to compute higher order differences
    Parameters
    ----------
    array : sparse array
    n : int, default: 1
        differencing order
    axis : int, default: -1
        axis along which differences are computed
    Returns
    -------
    diff_array : sparse array
                 same shape as input array,
                 but 'axis' dimension is smaller by 'n'.
    """
    if (n < 0) or (int(n) != n):
        raise ValueError('Expected order is non-negative integer, '
                         'but found: {}'.format(n))
    if not scipy.sparse.issparse(array):
        warnings.warn('Array is not sparse. Consider using numpy.diff')

    if n == 0:
        return array

    nd = array.ndim
    slice1 = [slice(None)]*nd
    slice2 = [slice(None)]*nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    A = sparse_diff(array, n-1, axis=axis)
    return A[slice1] - A[slice2]


def derivative(n, order=2):
    if n == 1:
        # no derivative for constant functions
        return scipy.sparse.csc_matrix(0.)
    D = sparse_diff(scipy.sparse.identity(n).tocsc(), n=order).tolil()
    return np.asarray(D.dot(D.T).tocsc().todense())


def generate_x_noise(n, x_noise):
    # with probability x_noise, there is uniform noise along attack region
    unit_noise = np.random.rand(n)
    unit_noise = (unit_noise < 0.5).astype(float)

    noise_mask = (np.random.rand(n) < x_noise).astype(float)
    return noise_mask * unit_noise * adv_eps


def get_feats(X, knots):
    X = X[:, np.newaxis]
    M = 4
    aug = np.arange(1, M)
    knots = np.r_[aug - M - knots[0], knots, aug + knots[-1]]

    bases = (X >= knots[:-1]).astype(np.int) * (X < knots[1:]).astype(np.int)
    # do recursion from Hastie et al. vectorized
    maxi = len(knots) - 1
    for m in range(2, M+1):
        maxi -= 1

        # left sub-basis
        num = (X - knots[:maxi])* bases[:, :maxi]
        denom = knots[m-1: maxi+m-1] - knots[:maxi]
        left = num/denom

        # right sub-basis
        num = (knots[m: maxi+m] - X) * bases[:, 1:maxi+1]
        denom = knots[m: maxi+m] - knots[1: maxi+1]
        right = num/denom

        bases = left + right
    return bases


def data_gen(n, weights):
    num_stairs = len(weights)
    # sample from categorical distribution
    X = np.random.choice(num_stairs, p=weights, size=n).astype(float)
    return X


def label_noise(n, eps):
    return eps * np.random.randn(n)


def sqerr(theta, feats, y):
    return np.square(feats.dot(theta) - y).sum()


def get_P(num_stairs):
    P = derivative(num_stairs*2 + 2, order=2)
    return P


def norm(theta, P):
    # return P.dot(theta[1:]).dot(theta[1:])
    return P.dot(theta).dot(theta)


def T(x):
    x_round = np.floor(x)
    return [x_round, x_round + adv_eps]


def sqerr_adv(theta, T_feats, y):
    sqerrs = [np.square(feats.dot(theta) - y) for feats in T_feats]
    max_errs = np.sum(np.maximum.reduce(sqerrs))
    return max_errs


def objective_adv(theta, lamda, T_feats, y, P):
    return sqerr_adv(theta, T_feats, y) + lamda*norm(theta, P)


def test_mse(theta, X_test, y_test, weights, noise_eps, x_noise, knots, slope, robust=False):
    def err_y_for_x(x, e_y):
        x_feats = get_feats(np.asarray([x]), knots)
        preds_x = x_feats.dot(theta)[0]
        err_for_pt = np.square(preds_x) - 2*preds_x*e_y + noise_eps**2 + np.square(e_y)
        return err_for_pt

    total_err = []
    for i in range(X_test.shape[0]):
        # otherwise we just use the number of stairs from X_test
        stair_i_err = 0
        candidates = [i + adv_eps]
        for x in candidates:
            if robust:
                stair_i_err = max(stair_i_err, err_y_for_x(x, slope*i))
            else:
                stair_i_err += x_noise * err_y_for_x(x, slope*i)
        if not robust:
            stair_i_err /= len(candidates)

        if robust:
            stair_i_err = max(stair_i_err, err_y_for_x(i, slope*i))
        else:
            stair_i_err += (1-x_noise)*err_y_for_x(i, slope*i)

        total_err.append(stair_i_err)
    total_err = np.asarray(total_err)
    return np.sum(total_err * weights)


def generate_dataset(num_examples, weights, noise_eps, x_noise, slope):
    X = data_gen(num_examples, weights)
    y = slope*X + label_noise(X.shape[0], noise_eps)
    X += generate_x_noise(num_examples, x_noise)
    num_unlabeled = 50000
    X_unlabeled = data_gen(num_unlabeled, weights)
    X_unlabeled += generate_x_noise(num_unlabeled, x_noise)
    X_unlabeled = np.sort(X_unlabeled)
    # filter out overlap with labeled data
    in_training_data = np.zeros(X_unlabeled.size).astype(bool)
    for x in set(X):
        index = np.searchsorted(X_unlabeled, x)
        if index >= 0:
            in_training_data |= (X_unlabeled == x)
    X_unlabeled = X_unlabeled[~in_training_data]
    return X, y, X_unlabeled


def get_test_set(num_stairs, x_noise, slope):
    # X_test has every row a different class and columns are samples from that class
    # y_test is a noiseless label
    num_samples_per_stair = 10000

    y_test = slope * np.arange(num_stairs).astype(float)[:, np.newaxis] * np.ones((num_stairs, num_samples_per_stair))
    X_test = np.arange(num_stairs).astype(float)[:, np.newaxis]
    # one sided stair
    noise = np.random.rand(num_stairs, num_samples_per_stair)*adv_eps

    X_test = X_test + noise
    return X_test, y_test


def generalization_gap(theta, feats, y, precomp_test_mse):
    return (precomp_test_mse - sqerr(theta, feats, y)/feats.shape[0])


def solve_minnorm(T_feats, y, P):
    theta_var = cp.Variable(T_feats[0].shape[1])
    constraints = [dat@theta_var == y for dat in T_feats]
    objective = cp.Minimize(cp.quad_form(theta_var, P))
    prob = cp.Problem(objective, constraints=constraints)
    try:
        prob.solve()
    except Exception:
        prob.solve(solver='SCS')
    return Result(x=theta_var.value)


def solve_rst(T_feats, y, T_u_feats, y_u):
    unlabeled, unlabeled_aug = T_u_feats[0], T_u_feats[1]
    theta_var = cp.Variable(T_feats[0].shape[1])
    constraints = [dat@theta_var == y for dat in T_feats]
    constraints += [unlabeled@theta_var == unlabeled_aug@theta_var]
    obj = cp.sum_squares(unlabeled@theta_var - y_u)
    objective = cp.Minimize(obj)
    prob = cp.Problem(objective, constraints=constraints)
    try:
        prob.solve(solver='OSQP', verbose=True, max_iter=5000, eps_abs=1e-10, eps_rel=1e-10)
    except Exception:
        try:
            prob.solve(solver='ECOS')
        except Exception:
            prob.solve(solver='SCS')

    print(prob.status)
    return Result(x=theta_var.value)


def solve_selftrain(T_feats, y, S_u, theta_std):
    theta_var = cp.Variable(T_feats[0].shape[1])
    constraints = [dat@theta_var == y for dat in T_feats]
    objective = cp.Minimize(cp.quad_form(theta_var-theta_std, S_u))
    prob = cp.Problem(objective, constraints=constraints)
    try:
        prob.solve()
    except Exception:
        try:
            prob.solve(solver='ECOS')
        except Exception:
            for i in range(10):
                prob.solve(solver='SCS')
                if theta_var.value is not None:
                    break
    return Result(x=theta_var.value)


def get_results(res, feats, y, X_test, y_test, weights, noise_eps, x_noise, knots, slope, P, T_feats):
    curr_test_mse = test_mse(res.x, X_test, y_test, weights, noise_eps, x_noise, knots, slope, robust=False),
    return {
        'Generalization gap': generalization_gap(res.x, feats, y, curr_test_mse),
        'Test MSE': curr_test_mse,
        'Norm': norm(res.x, P),
        'Training robust MSE': sqerr_adv(res.x, T_feats, y) / feats.shape[0],
        'Test robust MSE': test_mse(res.x, X_test, y_test, weights, noise_eps, x_noise, knots, slope, robust=True)}


def solve_for_X_y(X, y, X_u, lamda, P, X_test, y_test, weights, x_noise, num_stairs, save_dir, slope, plot=False):
    knots = np.r_[np.arange(num_stairs), np.arange(num_stairs)+adv_eps]
    knots = np.sort(knots)

    feats = get_feats(X, knots)

    curr_res_n = solve_minnorm([feats], y, P)

    # pre-generate the adversarial examples
    Xs = T(X)
    T_feats = [get_feats(dat, knots) for dat in Xs]

    # min norm estimator data augmentation
    curr_res_a = solve_minnorm(T_feats, y, P)

    # robust self training
    feats_unlabeled = get_feats(X_u, knots)
    T_u_feats = [get_feats(dat, knots) for dat in T(X_u)]
    y_u = feats_unlabeled @ curr_res_n.x
    curr_res_rst = solve_rst(T_feats, y, T_u_feats, y_u)

    # compute generalization gap
    res_dict_n = get_results(curr_res_n, feats, y, X_test, y_test, weights, noise_eps, x_noise, knots, slope, P, T_feats)
    res_dict_a = get_results(curr_res_a, feats, y, X_test, y_test, weights, noise_eps, x_noise, knots, slope, P, T_feats)
    res_dict_rst = get_results(curr_res_rst, feats, y, X_test, y_test, weights, noise_eps, x_noise, knots, slope, P, T_feats)

    return res_dict_n, res_dict_a, res_dict_rst


def solve_for_n_lamda(num_examples, lamda, X_y, X_test, y_test, P, weights, noise_eps, x_noise, num_stairs, save_dir, slope):
    res_normal = defaultdict(list)
    res_adv = defaultdict(list)
    res_rst = defaultdict(list)

    plot_bools = np.zeros(len(X_y)).astype(bool)
    plot_bools[-1] = True
    if args.debug:
        n_jobs = 1
    else:
        n_jobs = -2
    res = Parallel(n_jobs=n_jobs)(delayed(solve_for_X_y)(
        X, y, X_u, lamda, P, X_test, y_test, weights, x_noise, num_stairs, save_dir, slope, plot) for (X, y, X_u), plot in zip(X_y, plot_bools))
    for res_dict_n, res_dict_a, res_dict_rst in res:
        if res_dict_n is None or res_dict_a is None:
            continue

        for k, v in res_dict_n.items():
            res_normal[k].append(v)
        for k, v in res_dict_a.items():
            res_adv[k].append(v)
        for k, v in res_dict_rst.items():
            res_rst[k].append(v)

    agg_res_normal = {k: np.mean(v) for k, v in res_normal.items()}
    agg_res_adv = {k: np.mean(v) for k, v in res_adv.items()}
    agg_res_rst = {k: np.mean(v) for k, v in res_rst.items()}
    agg_res_normal_std = {k: np.std(v) / np.sqrt(len(v)-1) for k, v in res_normal.items()}
    agg_res_adv_std = {k: np.std(v) / np.sqrt(len(v)-1) for k, v in res_adv.items()}
    agg_res_rst_std = {k: np.std(v) / np.sqrt(len(v)-1) for k, v in res_rst.items()}
    return agg_res_normal, agg_res_adv, agg_res_rst, agg_res_normal_std, agg_res_adv_std, agg_res_rst_std


def populate_res(res, arr, i, j):
    for k, v in res.items():
        arr[k][i, j] = v


def get_num_examples(num_stairs):
    num_examples = [
        num_stairs*2+2, num_stairs*3,
        int(num_stairs*3.5), num_stairs*4, num_stairs*5, num_stairs*8,
        num_stairs*10, num_stairs*20, num_stairs*30, num_stairs*40,
        num_stairs*50, num_stairs*100]

    return np.asarray(num_examples)


def run_for_stair_count(num_stairs, noise_eps, x_noise, slope):
    if num_stairs < 5:
        raise ValueError("Number of stairs < 5 not supported")

    # uniform weights but more concentrated on the beginning part
    weights_1 = np.asarray([1/5]*5)
    weights_2 = np.asarray([0.01]*(num_stairs-5))
    weights = np.concatenate([weights_1, weights_2])
    weights /= np.sum(weights)

    num_examples = get_num_examples(num_stairs)
    num_ns = len(num_examples)

    X_test, y_test = get_test_set(num_stairs, x_noise, slope)

    # initialize P for the penalty
    P = get_P(num_stairs)

    # # make a matrix of lamdas vs num examples
    def create_empty():
        return {
            'Test MSE': np.empty((num_lamdas, num_ns)),
            'Generalization gap': np.empty((num_lamdas, num_ns)),
            'Norm': np.empty((num_lamdas, num_ns)),
            'Training robust MSE': np.empty((num_lamdas, num_ns)),
            'Test robust MSE': np.empty((num_lamdas, num_ns))}

    lamda_vs_n_normal = create_empty()
    lamda_vs_n_adv = create_empty()
    lamda_vs_n_rst = create_empty()
    lamda_vs_n_normal_std = create_empty()
    lamda_vs_n_adv_std = create_empty()
    lamda_vs_n_rst_std = create_empty()

    save_dir = f"stairs{num_stairs}_xnoise{args.x_noise}_slope{args.slope}"
    if args.less_trials:
        save_dir += "_lesstrials"
    save_dir = Path(save_dir).resolve().expanduser()
    save_dir.mkdir(exist_ok=False)

    for j, n in tqdm(enumerate(num_examples), total=len(num_examples)):
        if args.debug:
            num_trials = 2
        else:
            if args.less_trials:
                num_trials = 5 if j > 3 else 200
            else:
                num_trials = 25 if j > 8 else 2000

        X_y = [generate_dataset(n, weights, noise_eps, x_noise, slope) for _ in range(num_trials)]
        for i, lamda in tqdm(enumerate(lamdas), total=len(lamdas)):
            agg_res_normal, agg_res_adv, agg_res_rst, n_std, a_std, rst_std = solve_for_n_lamda(
                n, lamda, X_y, X_test, y_test, P, weights, noise_eps, x_noise, num_stairs, save_dir, slope)
            populate_res(agg_res_normal, lamda_vs_n_normal, i, j)
            populate_res(agg_res_adv, lamda_vs_n_adv, i, j)
            populate_res(agg_res_rst, lamda_vs_n_rst, i, j)
            populate_res(n_std, lamda_vs_n_normal_std, i, j)
            populate_res(a_std, lamda_vs_n_adv_std, i, j)
            populate_res(rst_std, lamda_vs_n_rst_std, i, j)

    # save the results
    with open(save_dir / f'splines_lamda_vs_n_normal.pkl', 'wb') as f:
        pickle.dump(lamda_vs_n_normal, f, pickle.HIGHEST_PROTOCOL)
    with open(save_dir / f'splines_lamda_vs_n_adv.pkl', 'wb') as f:
        pickle.dump(lamda_vs_n_adv, f, pickle.HIGHEST_PROTOCOL)
    with open(save_dir / f'splines_lamda_vs_n_rst.pkl', 'wb') as f:
        pickle.dump(lamda_vs_n_rst, f, pickle.HIGHEST_PROTOCOL)
    with open(save_dir / f'splines_lamda_vs_n_normal_std.pkl', 'wb') as f:
        pickle.dump(lamda_vs_n_normal_std, f, pickle.HIGHEST_PROTOCOL)
    with open(save_dir / f'splines_lamda_vs_n_adv_std.pkl', 'wb') as f:
        pickle.dump(lamda_vs_n_adv_std, f, pickle.HIGHEST_PROTOCOL)
    with open(save_dir / f'splines_lamda_vs_n_rst_std.pkl', 'wb') as f:
        pickle.dump(lamda_vs_n_rst_std, f, pickle.HIGHEST_PROTOCOL)


def set_ylim(normal_y, adv_y, percentile=95):
    # check if outlier situation is occurring
    normal_med = np.median(normal_y)
    adv_med = np.median(adv_y)
    normal_max_med_diff = np.amax(normal_y) - normal_med
    adv_max_med_diff = np.amax(adv_y) - adv_med

    ylim_min, ylim_max = None, None
    if normal_max_med_diff > normal_med*100 or adv_max_med_diff > adv_med*100:
        normal_95 = np.percentile(normal_y, percentile)
        adv_95 = np.percentile(adv_y, percentile)
        ylim_max = max(normal_95, adv_95)

    if not (ylim_min is None and ylim_max is None):
        plt.ylim([ylim_min, ylim_max])

def make_diff_plot(dirs, rst=False, save=False, newfig=True, color_idx=None, marker_idx=None, label_prefix="", k='Test MSE'):
    save_dirs = [Path(d) for d in dirs]

    num_stairss = [10]
    lamda_idxs = {0}
    sizes = [(5,4), (5,4.3)]

    # make diff plot over lamdas per stair
    for save_dir, num_stairs, size in zip(save_dirs, num_stairss, sizes):
        with open(save_dir / f'splines_lamda_vs_n_normal.pkl', 'rb') as f:
            lamda_vs_n_normal = pickle.load(f)
        with open(save_dir / f'splines_lamda_vs_n_normal_std.pkl', 'rb') as f:
            lamda_vs_n_normal_std = pickle.load(f)

        if rst:
            with open(save_dir / f'splines_lamda_vs_n_rst.pkl', 'rb') as f:
                lamda_vs_n_adv = pickle.load(f)
            with open(save_dir / f'splines_lamda_vs_n_rst_std.pkl', 'rb') as f:
                lamda_vs_n_adv_std = pickle.load(f)
        else:
            with open(save_dir / f'splines_lamda_vs_n_adv.pkl', 'rb') as f:
                lamda_vs_n_adv = pickle.load(f)
            with open(save_dir / f'splines_lamda_vs_n_adv_std.pkl', 'rb') as f:
                lamda_vs_n_adv_std = pickle.load(f)
        if newfig:
            plt.figure(figsize=size)

        num_examples = get_num_examples(num_stairs)
        all_normal_ys = []
        all_adv_ys = []
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        colors = colors[2:] + colors[:2] + ['lightseagreen', 'mediumseagreen', 'seagreen', 'purple', 'mediumaquamarine']
        markers = ['o', '^', 's', 'd']
        if color_idx is None:
            color = colors[0]
        else:
            color = colors[color_idx]
        if marker_idx is None:
            marker = markers[0]
        else:
            marker = markers[marker_idx]

        # choose best lambda per n
        normal_y_idxs = np.argmin(lamda_vs_n_normal['Test MSE'], axis=0)
        adv_y_idxs = np.argmin(lamda_vs_n_adv['Test MSE'], axis=0)
        normal_y = lamda_vs_n_normal[k][normal_y_idxs, np.arange(normal_y_idxs.size)]
        adv_y = lamda_vs_n_adv[k][adv_y_idxs, np.arange(adv_y_idxs.size)]
        normal_std_y = lamda_vs_n_normal_std[k][normal_y_idxs, np.arange(normal_y_idxs.size)]
        adv_std_y = lamda_vs_n_adv_std[k][adv_y_idxs, np.arange(adv_y_idxs.size)]

        all_normal_ys.append(normal_y)
        all_adv_ys.append(adv_y)

        # plot the diff plots
        diff = adv_y - normal_y
        std = np.sqrt(np.square(normal_std_y) + np.square(adv_std_y))
        num_examples_ = num_examples

        # mask
        mask = (num_examples_ > (3 * num_stairs + 2))
        num_examples_ = num_examples_[mask]
        diff = diff[mask]
        std = std[mask]
        if not label_prefix:
            plt.semilogx(num_examples_, diff,
                 color=color, marker=marker, markersize=10)
        else:
            label = label_prefix
            plt.semilogx(num_examples_, diff,
                         label=label_prefix,
                         color=color, marker=marker, markersize=10)

        plt.fill_between(
            num_examples_,
            diff - std,
            diff + std,
            alpha=0.2, color=color)

        plt.semilogx(num_examples_, np.zeros(len(num_examples_)), linestyle='dashed', color='gray')
        plt.xlabel('Number of labeled samples')
        plt.ylabel(f'Test Err(Aug) - Test Err(Std)')

        all_normal_ys = np.concatenate(all_normal_ys)
        all_adv_ys = np.concatenate(all_adv_ys)
        set_ylim(all_normal_ys, all_adv_ys, 95)

        if rst:
            plt.legend()
        plt.tight_layout()

def make_plot():
    '''
    Recreates Figure 6a
    '''
    plt.rcParams.update({'font.size': 17})
    save_dir = "stairs10_xnoise0.01_slope1"
    if args.less_trials:
        save_dir += '_lesstrials'
    save_dirs = [Path(save_dir)]

    make_diff_plot(save_dirs, rst=False, save=False, newfig=True, color_idx=9, marker_idx=0, label_prefix="Augmented")
    make_diff_plot(save_dirs, rst=True, save=False, newfig=False, color_idx=14, marker_idx=1, label_prefix="RST")
    plt.tight_layout()
    plt.savefig('spline_tradeoff.png', bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Splines')
    parser.add_argument('--num_stairs', type=int, default=10,
                        help="Number of stairs")
    parser.add_argument('--x_noise', type=float, default=0.01,
                        help="probability of sampling from ball")
    parser.add_argument('--slope', type=float, default=1,
                        help="slope of y=mx")
    parser.add_argument('--debug', action='store_true', default=False,
                        help="run a small amount")
    parser.add_argument('--less_trials', action='store_true', default=False,
                        help="less trials")
    args = parser.parse_args()

    np.random.seed(111)

    adv_eps = 1.0 / 2
    x_noise = args.x_noise
    slope = args.slope
    # noiseless
    noise_eps = 0

    ext = '.png'
    # lambda not used
    num_lamdas = 1
    lamdas = np.asarray([0])

    run_for_stair_count(args.num_stairs, noise_eps, x_noise, slope)
    make_plot()
