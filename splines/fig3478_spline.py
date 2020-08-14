import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from fig2_spline import get_feats, get_P

matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = matplotlib.patches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height, color='black')
    return p


def plot_augmentations_polar(S0, theta_star):
    angles = np.r_[np.linspace(0, 2*np.pi, 1000), np.pi, np.pi/2, 3*np.pi/2]
    fig, ax = plt.subplots(figsize=(6, 6))
    bad_c = 'white'
    good_c = 'C1'

    ax.set_facecolor(bad_c)

    def get_lhs_rhs(x):
        rho = (np.inner(x, theta_star)) / np.maximum(np.inner(x, x), 1e-16)
        lhs = rho**2 * S0.dot(x).dot(x)
        rhs = 2 * rho * S0.dot(x).dot(theta_star)
        return lhs, rhs

    for angle in angles:
        x1v = 2*np.pi * np.cos(angle)
        x2v = 2*np.pi * np.sin(angle)
        x = np.asarray([x1v, x2v])
        lhs, rhs = get_lhs_rhs(x)

        if rhs - lhs > -1e-10:
            plt.plot([0, x1v], [0, x2v], color=good_c, zorder=0)
    plt.plot([0],[0], color=good_c, zorder=0)
    plt.axis([-1, 1, -1, 1])
    arrow = plt.arrow(0, 0, theta_star[0], theta_star[1], width=0.02, alpha=1, length_includes_head=True, color='black', zorder=100)
    plt.legend([arrow],
               [r'$\theta^*$'],
               loc="lower right",
               handler_map={matplotlib.patches.FancyArrow : matplotlib.legend_handler.HandlerPatch(patch_func=make_legend_arrow),})
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.text(0, 1.1, '$e_2$', horizontalalignment='center')
    ax.text(1.1, 0, '$e_1$',  verticalalignment='center')


##################
# Figure 3
##################
matplotlib.rcParams.update({'font.size': 28})
theta_star = np.asarray([1, 0.2])
X = np.diag([1, 2])
S0 = X.T @ X
plot_augmentations_polar(S0, theta_star)
plt.subplots_adjust(bottom=0.15, left=0.2)
plt.savefig('flag_less_skew.png')

matplotlib.rcParams.update({'font.size': 28})
theta_star = np.asarray([1, 0.2])
X = np.diag([1, 5])
S0 = X.T @ X
plot_augmentations_polar(S0, theta_star)
plt.subplots_adjust(bottom=0.15, left=0.2)
plt.savefig('flag_more_skew.png')


##################
# Figure 7
##################

num_stairs = 10
num_examples = 22
adv_eps = (1.0 / 2)
noise_eps = 0.0
x_noise = 0.1
slope = 1
np.set_printoptions(precision=5)
discrete_support = True

knots = np.r_[np.arange(num_stairs), np.arange(num_stairs)+adv_eps]
knots = np.sort(knots)
weights_1 = np.asarray([1/5]*5)
weights_2 = np.asarray([0.01]*(num_stairs-5))
weights = np.concatenate([weights_1, weights_2])
weights /= np.sum(weights)
X = np.r_[np.arange(5).astype(float)]
X = np.sort(X)
y = slope*np.floor(X)

# compute the population \Sigma_0

# first we must rotate the spline basis in a way that the correct norm is being minimized
feats = get_feats(X, knots)
# add small identity for numerical stability
P = get_P(knots) + 1e-10 * np.eye(22)
eigvals, eigs = np.linalg.eig(P)
eigvals = np.maximum(eigvals, 0)
Q = eigs.dot(np.linalg.pinv(np.diag(np.sqrt(eigvals)))).dot(eigs.T)
P_half = np.linalg.inv(Q)
# Q.T X^T X Q

S0_trans = np.zeros((feats.shape[1], feats.shape[1]))
for x in range(num_stairs):
    x1, x2 = get_feats(np.asarray([x, x+adv_eps]), knots).dot(Q)
    S0_trans += (1 - x_noise) * weights[x] * np.outer(x1, x1) + x_noise * weights[x] * np.outer(x2, x2)

def solve_rotated(X, y):
    feats = get_feats(X, knots)
    feats_trans = feats.dot(Q)
    theta_trans = np.linalg.pinv(feats_trans.T.dot(feats_trans)).dot(feats_trans.T.dot(y))
    return feats_trans, theta_trans

feats_std_trans, theta_std_trans = solve_rotated(X, y)
# construct theta_star
all_xs = np.r_[np.asarray([i for i in range(num_stairs)]), np.asarray([i + adv_eps for i in range(num_stairs)])]
all_xs = np.sort(all_xs)
all_ys = slope*np.floor(all_xs)
all_feats_trans, theta_star_trans = solve_rotated(all_xs, all_ys)

def plot_std_aug(theta_std, theta_aug):
    X_stairs = np.arange(0, num_stairs).astype(float)
    y_stairs = slope*X_stairs
    for X_stair, y_stair in zip(X_stairs, y_stairs):
        plt.plot([X_stair, X_stair+adv_eps], [y_stair, y_stair], color='black', alpha=0.5)

    X_t = np.linspace(0, num_stairs-0.5, 100)
    plt.plot(X_t, get_feats(X_t, knots).dot(Q).dot(theta_std), label='Standard', linestyle='dashed', lw=5)
    plt.plot(X_t, get_feats(X_t, knots).dot(Q).dot(theta_aug), label='Augmented', linestyle='solid', lw=5)
    plt.legend()
    plt.legend()
    plt.scatter(X, y, color='black', s=75, zorder=1000)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$f_{\theta}(t)$')

# add 3.5
matplotlib.rcParams.update({'font.size': 18})

X_aug = np.r_[X, 3.5]
X_aug = np.sort(X_aug)
y_aug = slope*np.floor(X_aug)
feats_aug, theta_aug = solve_rotated(X_aug, y_aug)
plt.figure(figsize=(5,5))

plot_std_aug(theta_std_trans, theta_aug)
plt.axis('equal')
plt.xlim([-0.5, 10])
plt.ylim([-0.5, 10])
plt.xticks(np.arange(0, 10, 2.0))
plt.yticks(np.arange(0, 10, 2.0))
plt.scatter([3.5], [3], marker='X', s=75, color='C2', zorder=1000)

plt.savefig('spline_add_35.png')

# add 4.5
matplotlib.rcParams.update({'font.size': 22})
feats_std_trans, theta_std_trans = solve_rotated(X, y)
all_xs = np.r_[np.asarray([i for i in range(num_stairs)]), np.asarray([i + adv_eps for i in range(num_stairs)])]
all_xs = np.sort(all_xs)
all_ys = slope*np.floor(all_xs)
all_feats_trans, theta_star_trans = solve_rotated(all_xs, all_ys)# add 4.5
matplotlib.rcParams.update({'font.size': 18})
X_aug = np.r_[X, 4.5]
X_aug = np.sort(X_aug)
y_aug = slope*np.floor(X_aug)
feats_aug, theta_aug = solve_rotated(X_aug, y_aug)
plt.figure(figsize=(5,5))

plot_std_aug(theta_std_trans, theta_aug)
plt.axis('equal')
plt.xlim([-0.5, 10])
plt.ylim([-0.5, 10])
plt.xticks(np.arange(0, 10, 2.0))
plt.yticks(np.arange(0, 10, 2.0))
plt.scatter([4.5], [4], marker='X', s=75, color='C2', zorder=1000)
plt.savefig('spline_add_45.png')

# plot the difference in test error as suggested in Theorem 1, Fig 7a

plt.clf()
# check if a perturbation does/does not satisfy the criterion
hatS0 = feats_std_trans.T.dot(feats_std_trans)

def proj(S, rank_S=None):
    eigvals, eigs = np.linalg.eig(S)
    if rank_S is not None:
        sort_idx = np.argsort(-eigvals)
        eigvals[sort_idx[:rank_S]] = 1
        eigvals[sort_idx[rank_S:]] = 0
    else:
        eigvals[eigvals <= 1e-8] = 0.0
        eigvals[eigvals > 0] = 1.0
    return eigs.dot(np.diag(eigvals)).dot(eigs.T).real
hat_proj_0 = np.eye(hatS0.shape[0]) - proj(hatS0, rank_S=feats_std_trans.shape[0])

def criterion(S0, theta_star, proj0, x):
    theta_0 = proj0.dot(theta_star)
    u = proj0.dot(x)
    if np.inner(u, u) < 1e-10:
        return 0
    rho = np.inner(theta_0, u) / np.inner(u, u)
    diff = 2 * rho * S0.dot(theta_0).dot(u) - rho**2 * S0.dot(u).dot(u)
    return diff

matplotlib.rcParams.update({'font.size': 16})

# on the line
lines = np.arange(10).astype(float)
line_feats = get_feats(lines, knots)
line_feats_trans = line_feats.dot(Q)
line_diffs = []
for i in range(line_feats_trans.shape[0]):
    x = line_feats_trans[i]
    diff = -criterion(S0_trans, theta_star_trans, hat_proj_0, x).real
    line_diffs.append(diff)
# not on the line
perts = np.arange(10).astype(float) + adv_eps
pert_feats = get_feats(perts, knots)
pert_feats_trans = pert_feats.dot(Q)
pert_diffs = []
for i in range(pert_feats_trans.shape[0]):
    x = pert_feats_trans[i]
    diff = -criterion(S0_trans, theta_star_trans, hat_proj_0, x).real
    pert_diffs.append(diff)
plt.scatter(lines, line_diffs, label='On the line', marker='o', s=90)
plt.scatter(perts, pert_diffs, label='Perturbations', marker='^', s=90)
plt.ylabel('Bias criterion (Aug - Std)')
plt.xlabel(r'Augmentation point ($t$)')
plt.xticks(np.arange(0, 10, 1.0))
plt.legend(loc="upper right")
plt.subplots_adjust(bottom=0.15, left=0.15)
plt.savefig('spline_perturbations.png')
matplotlib.rcParams.update({'font.size': 22})

###############
# Fig 4
###############
matplotlib.rcParams.update({'font.size': 28})
# curr dataset
X = np.r_[0,1]
X = np.sort(X)
y = slope*np.floor(X)

# rotation matrix
P = get_P(knots) + 1e-10 * np.eye(22)
eigvals, eigs = np.linalg.eig(P)
eigvals = np.maximum(eigvals, 0)
Q = eigs.dot(np.linalg.pinv(np.diag(np.sqrt(eigvals)))).dot(eigs.T)

X0 = get_feats(X, knots).dot(Q)

xaug_raw = np.r_[X, 4.5]
Xaug = get_feats(xaug_raw, knots).dot(Q)
yaug = np.floor(xaug_raw)

# std estimator
stdest = np.linalg.pinv(X0.T @ X0) @ (X0.T @ y)

augest = np.linalg.pinv(Xaug.T @ Xaug) @ (Xaug.T @ yaug)
# sigma
S_trans = all_feats_trans.T @ all_feats_trans
S_eigs, S_eigv = np.linalg.eig(S_trans)

for i in range(S_eigv.shape[1]):
    if i > 5:
        break
    plt.figure()
    plt.plot(np.arange(S_eigv.shape[0]), S_eigv[:, i], lw=5)
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.title('$q_{%d}$' % (i+1))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'eig{i}.png')


##########
# Fig 8
##########

matplotlib.rcParams.update({'font.size': 23})

def normalize(x):
    return x / np.linalg.norm(x)
plt.figure(figsize=(5,5))
fig, ax = plt.subplots()
local_idx = 19
global_idx = 2

val_8 = 5
val_45 = 1.5
xaug8 = get_feats(np.asarray([val_8]), knots).dot(Q).squeeze()
xaug45 = get_feats(np.asarray([val_45]), knots).dot(Q).squeeze()
projX0 = np.eye(X0.shape[1]) - proj(X0.T @ X0, rank_S=X0.shape[0])
S_gl = (S_eigv.T @ projX0)[[local_idx, global_idx]]
tstar_proj = normalize(S_gl @ theta_star_trans)
# we reflected these two vectors for better presentation
xaug_proj8 = -normalize( S_gl @ xaug8)
xaug_proj45 = -normalize(S_gl @ xaug45)

# plotting code
text_eps = 0.1
arr1 = ax.arrow(0, 0, tstar_proj[0], tstar_proj[1], head_width=0.05, head_length=0.1, length_includes_head=True, lw=3, color='C0', zorder=100)
ax.text(tstar_proj[0]-0.1, tstar_proj[1] - 4*text_eps, r'$\theta^* - \hat{\theta}_{std}$', usetex=True, color='C0')
arr2 = ax.arrow(0, 0, xaug_proj8[0] , xaug_proj8[1], head_width=0.05, head_length=0.1, length_includes_head=True, lw=3, color='C4', zorder=100)
ax.text(xaug_proj8[0]-10*text_eps, xaug_proj8[1], r'$\Pi_{lg}X(5)$', usetex=True, color='C4')
arr3 = ax.arrow(0, 0, xaug_proj45[0], xaug_proj45[1], head_width=0.05, head_length=0.1, length_includes_head=True, lw=3, color='C1', zorder=100)
ax.text(xaug_proj45[0]+text_eps, xaug_proj45[1]-0.1, r'$\Pi_{lg}X(1.5)$', usetex=True, color='C1')

tstar_proj_proj = np.inner(tstar_proj, xaug_proj45) * xaug_proj45
plt.plot([1, tstar_proj_proj[0]], [0, tstar_proj_proj[1]], linestyle='dashed', color='C2')

slope_perp = (-tstar_proj_proj[1]) / (1-tstar_proj_proj[0])
little_sq_x_int = 0.75
b_perp = -(slope_perp * little_sq_x_int)
plt.plot([0.3, 0.2], [0.3*slope_perp + b_perp, 0.2*slope_perp + b_perp], linestyle='solid', color='C2')
slope_nonperp = xaug_proj45[1] / xaug_proj45[0]
b_nonperp = -0.15*slope_nonperp
plt.plot([0.3, 0.36], [0.3*slope_nonperp + b_nonperp, 0.36*slope_nonperp + b_nonperp], linestyle='solid', color='C2')

ax.set_ylim([-1, 1.4])
ax.set_xlim([-0.2, 1.5])
ax.set_aspect('equal')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.set_yticklabels([])
ax.set_xticklabels([])
ax.text(0, 1.6, 'Global ($q_3$)', horizontalalignment='center', weight='bold')
ax.text(1.9, 0, 'Local ($q_{2s}$)',  verticalalignment='center', weight='bold')
# remove the ticks from the top and right edges

plt.tight_layout()
plt.savefig('local_global.png')
