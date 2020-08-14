'''
Generates Figure 2: Spline augmentation and RST visualization
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy as scipy
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams.update({'font.size': 22})
import cvxpy as cp

save = True


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

def get_D(n, order=2):
    D = sparse_diff(scipy.sparse.identity(n).tocsc(), n=order).tolil()
    return np.asarray(D.tocsc().todense())

def get_P(knots, with_intercept=False):
    P = derivative(len(knots) + 2, order=2)
    return P

def R(x, z):
    return (np.power(z-0.5, 2)-1/12)*(np.power(x-1/2,2)-1/12)/4 - (np.power(np.abs(x-z)-1/2, 4) - 0.5*np.power(np.abs(x-z)-1/2, 2)+7/240) / 24

def get_feats(X, knots):
    X = X[:, np.newaxis]
    M = 4
    aug = np.arange(1, M)
    knots = np.r_[aug - M - 1 - knots[0], knots, aug + knots[-1]]

    K = len(knots)
    bases = (X >= knots[:-1]).astype(np.int) * (X < knots[1:]).astype(np.int)
    # do recursion from Hastie et al. vectorized
    maxi = len(knots) - 1
    for m in range(2, M+1):
        maxi -= 1

        # left sub-basis
        num = (X - knots[:maxi])* bases[:, :maxi]
        denom = knots[m-1 : maxi+m-1] - knots[:maxi]
        left = num/denom

        # right sub-basis
        num = (knots[m : maxi+m] - X) * bases[:, 1:maxi+1]
        denom = knots[m:maxi+m] - knots[1 : maxi+1]
        right = num/denom

        bases = left + right
    return bases

def solve_minnorm(T_feats, y, P):
    theta_var = cp.Variable(T_feats[0].shape[1])
    constraints = [dat@theta_var == y for dat in T_feats]
    objective = cp.Minimize(cp.quad_form(theta_var, P))
    prob = cp.Problem(objective, constraints=constraints)
    try:
        prob.solve(solver='OSQP', verbose=True, max_iter=5000, eps_abs=1e-10, eps_rel=1e-10)
    except Exception:
        prob.solve(solver='ECOS', verbose=True15)
    return theta_var.value

def solve_rst(T_feats, y, T_u_feats, y_u):
    unlabeled, unlabeled_aug = T_u_feats[0], T_u_feats[1]
    theta_var = cp.Variable(T_feats[0].shape[1])
    constraints = [dat@theta_var == y for dat in T_feats]
    constraints += [unlabeled@theta_var == unlabeled_aug@theta_var]
    obj = cp.sum_squares(unlabeled@theta_var - y_u)
    objective = cp.Minimize(obj)
    prob = cp.Problem(objective, constraints=constraints)
    prob.solve(solver='OSQP', verbose=True, max_iter=5000, eps_abs=1e-10, eps_rel=1e-10)

    print(prob.status)
    return theta_var.value

adv_eps = (1.0 / 2)
noise_eps = 0.0
x_noise = 0.1
slope = 1
np.set_printoptions(precision=5)

matplotlib.rcParams.update({'font.size': 25})

num_stairs = 7
discrete_support = True

def generate_x_noise(n):
    # probability x_noise there is noise
    if discrete_support:
        unit_noise = (np.random.rand(n) < 0.5).astype(float)
    else:
        unit_noise = np.random.rand(n)
    noise_mask = (np.random.rand(n) < x_noise).astype(float)
    return noise_mask * unit_noise*adv_eps

def data_gen(n, weights):
    num_stairs = len(weights)
    # sample from categorical distribution
    X = np.random.choice(num_stairs, p=weights, size=n).astype(float)
    return X

knots = np.r_[np.arange(num_stairs), np.arange(num_stairs)+adv_eps]
knots = np.sort(knots)

weights_1 = np.asarray([1/5]*5)
weights_2 = np.asarray([0.01]*(num_stairs-5))
weights = np.concatenate([weights_1, weights_2])
weights /= np.sum(weights)

P = get_P(knots)
X = np.r_[np.arange(4).astype(float)]
X = np.sort(X)
y = slope*np.floor(X)
feats = get_feats(X, knots)
T_feats = [get_feats(dat, knots) for dat in [np.floor(X), np.floor(X)+adv_eps]]
theta_std = solve_minnorm([feats], y, P)
theta_aug = solve_minnorm(T_feats, y, P)

X_u = data_gen(10000, weights)
X_u += generate_x_noise(10000)
T_u_feats = [get_feats(np.floor(X_u), knots), get_feats(np.floor(X_u)+adv_eps, knots)]
y_u = T_u_feats[0]@theta_std
theta_rst = solve_rst(T_feats, y, T_u_feats, y_u)


def plot_std_aug(theta_std, theta_aug, aug_label='Aug'):
    X_stairs = np.arange(0, num_stairs).astype(float)
    y_stairs = slope*X_stairs
    X_t = np.linspace(0, num_stairs-0.5, 100)
    plt.plot(X_t, get_feats(X_t, knots).dot(theta_std), label='Std', linestyle='dashed', lw=5)
    plt.plot(X_t, get_feats(X_t, knots).dot(theta_aug), label=aug_label, linestyle='solid', lw=5)
    line = plt.scatter(X, y, color='purple', zorder=100, s=80)
    perts = plt.scatter(np.floor(X) + adv_eps, y, color='C2', marker='x', zorder=100, s=80, linewidth=3)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$f_{\theta}(t)$')
    first_legend = plt.legend(prop={'size': 17}, loc='upper left')
    plt.gca().add_artist(first_legend)
    plt.legend(handles=[line, perts], labels=['$X_{std}$', '$X_{ext}$'],
               loc='upper left', prop={'size': 17}, bbox_to_anchor=(0, 0.66))

# Plot std and RST

plt.figure(figsize=(5.8,4))
plot_std_aug(theta_std, theta_rst, aug_label='RST')
plt.tight_layout()
plt.axis('equal')
plt.xticks([0,2,4])
plt.yticks([0,2,4])
plt.xlim([-0.1, num_stairs-1])
plt.ylim([-0.3, num_stairs-1])

if save:
    plt.savefig('spline_rst.png', bbox_inches='tight')

# Plot std and Aug

plt.figure(figsize=(5.8,4))
plot_std_aug(theta_std, theta_aug)
plt.tight_layout()
plt.axis('equal')
plt.xticks([0,2,4])
plt.yticks([0,2,4])
plt.xlim([-0.1, num_stairs-1])
plt.ylim([-0.3, num_stairs-1])
if save:
    plt.savefig('minnorm_aug.png', bbox_inches='tight')
