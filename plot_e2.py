import numpy as np
import matplotlib.pyplot as plt
import config
from scipy.ndimage import gaussian_filter1d
from strlearn.utils import scores_to_cummean

weights = config.str_weights()
weigths_names = config.str_weights_names()
borders = config.borders()
criteria = config.criteria()
base_clfs = config.base_clfs()
base_clfs_names = config.base_clf_names()
reps = 10
chunks = 500

pe=3

# print(borders)
# exit()

res = np.load('results/res_e1_all.npy')
raw_clfs_res = res[:,:,:len(base_clfs)]
mean_raw_clfs_res = np.mean(raw_clfs_res, axis=0)[:,:,:,0] # (streams x clfs x chunks)

meta_res = res[:,:,len(base_clfs):]
meta_res = meta_res.reshape((reps,len(weights),len(base_clfs),len(criteria),len(borders),pe,chunks-1))
# (reps x streams x base_clfs x criteria x  borders x (mean, prev, dsca) x chunks-1)

mean_meta_res = np.mean(meta_res, axis=0)
# (streams x base_clfs x criteria x borders x (mean, dsca) x chunks)

mean_mean = mean_meta_res[:,:,:,:,0]
mean_prev = mean_meta_res[:,:,:,:,1]
mean_dsca = mean_meta_res[:,:,:,:,2]
# (streams x base_clfs x criteria x borders)

for bc_id, bc in enumerate(base_clfs):

    # plot
    sig = 5

    plt.clf()
    fig, axx = plt.subplots(3,3,figsize=(8*1.618, 8),sharex=True, sharey=True)
    fig.suptitle("%s" % (base_clfs_names[bc_id]), fontsize=14)

    axx = axx.ravel()

    #raw clf res
    raw_res = mean_raw_clfs_res[:,bc_id]

    for w_id, w in enumerate(weights):
        ax = axx[w_id]

        ax.set_title(weigths_names[w_id])
        # ax.set_ylim(.5,1)
        ax.set_ylabel('BAC')
        ax.set_xlabel('chunk')

        a = scores_to_cummean(raw_res[w_id].reshape(1,chunks-1,1))
        b = scores_to_cummean(mean_mean[w_id,bc_id,0,5].reshape(1,chunks-1,1))
        c = scores_to_cummean(mean_dsca[w_id,bc_id,0,-1].reshape(1, chunks-1, 1))
        d = scores_to_cummean(mean_prev[w_id,bc_id,0,5].reshape(1, chunks-1, 1))

        # ax.plot(gaussian_filter1d(raw_res[w_id], sig), label='base clf', color='orange')
        # ax.plot(gaussian_filter1d(mean_mean[w_id,bc_id,0,5],sig), ls='--', label='MEAN', c='tomato')
        # ax.plot(gaussian_filter1d(mean_dsca[w_id,bc_id,0,-1],sig), ls='--', label='DSCA', c='dodgerblue')

        ax.plot(a[0,:,0], label='base clf', color='orange')
        ax.plot(b[0,:,0], ls='--', label='MEAN', c='tomato')
        ax.plot(c[0,:,0], ls='--', label='DSCA', c='dodgerblue')
        ax.plot(d[0,:,0], ls='--', label='PREV', c='forestgreen')


        ax.set_ylim(.5,1)
        ax.set_xlim(0, 500)
        ax.grid(ls=":")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.legend()
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig('foo.png')
    plt.savefig('figures/e2_%s.png' % base_clfs_names[bc_id])
