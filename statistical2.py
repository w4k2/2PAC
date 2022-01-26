import numpy as np
import matplotlib.pyplot as plt
import config
from scipy.ndimage import gaussian_filter1d
from strlearn.utils import scores_to_cummean
from tabulate import tabulate
from scipy import stats


def t_test_corrected(a, b, J=1, k=10):
    """
    Corrected t-test for repeated cross-validation.
    input, two 2d arrays. Repetitions x folds
    As default for 5x5CV
    """
    if J*k != a.shape[0]:
        raise Exception('%i scores received, but J=%i, k=%i (J*k=%i)' % (
            a.shape[0], J, k, J*k
        ))

    d = a - b
    bar_d = np.mean(d)
    bar_sigma_2 = np.var(d.reshape(-1), ddof=1)
    bar_sigma_2_mod = (1 / (J * k) + 1 / (k - 1)) * bar_sigma_2
    t_stat = bar_d / np.sqrt(bar_sigma_2_mod)
    pval = stats.t.sf(np.abs(t_stat), (k * J) - 1) * 2
    return t_stat, pval

weights = config.str_weights()
weigths_names = config.str_weights_names()
borders = config.borders()
criteria = config.criteria()
base_clfs = config.base_clfs()
base_clfs_names = config.base_clf_names()
methods = ["RAW", "MEAN", "PREV", "DSCA"]
reps = 10
chunks = 500

alpha = .05

wn = np.array(weigths_names)

res = np.load('results/res_e1_all.npy')
raw_clfs_res = res[:,:,:len(base_clfs)].squeeze()
print(raw_clfs_res.shape)
# streams x clfs, reps
mean_raw_clfs_res = np.mean(raw_clfs_res, axis=3)
mean_raw_clfs_res = np.swapaxes(mean_raw_clfs_res, 0, 1)
mean_raw_clfs_res = np.swapaxes(mean_raw_clfs_res, 1, 2)

meta_res = res[:,:,len(base_clfs):]
# (reps x streams x base_clfs x criteria x  borders x (mean, prev, dsca) x chunks)
meta_res = meta_res.reshape((reps,len(weights),len(base_clfs),len(criteria),len(borders),3,chunks-1))
# (reps x streams x base_clfs x criteria x  borders x (mean, prev, dsca))
mean_meta_res = np.mean(meta_res, axis=6)
# (reps x streams x base_clfs)
mean_mean = mean_meta_res[:,:,:,0,5,0]
mean_prev = mean_meta_res[:,:,:,0,-1,1]
mean_dsca = mean_meta_res[:,:,:,0,-1,2]

# streams x base_clfs x method x reps
scores = np.stack((mean_mean, mean_prev, mean_dsca), axis=3)
scores = np.swapaxes(scores, 0, 3)
scores = np.swapaxes(scores, 0, 1)
scores = np.swapaxes(scores, 1, 2)
print(scores.shape)
mean_scores = np.mean(scores, axis=3)

scores = np.concatenate((mean_raw_clfs_res[:, :, np.newaxis, :], scores), axis=2)
mean_scores = np.mean(scores, axis=3)

headers = ["STREAM"] + methods
all = []
for base_idx, base in enumerate(base_clfs_names):
    print(base)
    t = []
    for s_idx, stream in enumerate(weigths_names):
        t.append(["%s" % stream] +
                 ["%.3f" % v for v in mean_scores[s_idx, base_idx, :]])

        T, p = np.array(
            [[t_test_corrected(scores[s_idx, base_idx, i, :],
                   scores[s_idx, base_idx, j, :])
              for i in range(4)]
             for j in range(4)]
            ).swapaxes(0, 2)

        _ = np.where((p < alpha) * (T > 0))
        conclusions = [list(1 + _[1][_[0] == i])
                       for i in range(4)]

        t.append([''] + [", ".join(["%i" % i for i in c])
                         if len(c) > 0 and len(c) < 4-1
                         else ("all" if len(c) == 4-1 else "---")
                         for c in conclusions])

    print(tabulate(t, headers=headers))
    all.append(np.array(t))
    # exit()
print(all[0].shape)


for idx, table in enumerate(all):
    if idx == 0:
        whole = table
    else:
        whole = np.concatenate((whole, table[:, 1:]), axis=1)

headers = ["STREAM"] + 5 * methods
print(tabulate(whole, headers=headers, tablefmt="latex_booktabs"))
print(whole.shape)
