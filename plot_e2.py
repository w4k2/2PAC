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

wn = np.array(weigths_names)[[2,5,8]]
pe=3

# print(borders[5]) #28%
# exit()

res = np.load('results/res_e1_all.npy')
raw_clfs_res = res[:,:,:len(base_clfs)]
mean_raw_clfs_res = np.mean(raw_clfs_res, axis=0)[:,:,:,0] # (streams x clfs x chunks)

meta_res = res[:,:,len(base_clfs):]
meta_res = meta_res.reshape((reps,len(weights),len(base_clfs),len(criteria),len(borders),pe,chunks-1))
# (reps x streams x base_clfs x criteria x  borders x (mean, prev, dsca) x chunks-1)

mean_meta_res = np.mean(meta_res, axis=0)
# (streams x base_clfs x criteria x borders x (mean, prev, dsca) x chunks)

mean_mean = mean_meta_res[[2,5,8],:,:,:,0]
mean_prev = mean_meta_res[[2,5,8],:,:,:,1]
mean_dsca = mean_meta_res[[2,5,8],:,:,:,2]
# (streams x base_clfs x criteria x borders)

sig = 5

fig, axx = plt.subplots(3,5,figsize=(15, 7), sharex=True, sharey=True)

for bc_id, bc in enumerate(base_clfs):
    axx[0,bc_id].set_title(base_clfs_names[bc_id])
    axx[2,bc_id].set_xlabel('chunk')

    #raw clf res
    raw_res = mean_raw_clfs_res[[2,5,8],bc_id]
    for w_id in range(3):
        ax = axx[w_id, bc_id]
        axx[w_id,0].set_ylabel(wn[w_id])

        a = scores_to_cummean(raw_res[w_id].reshape(1,chunks-1,1))
        b = scores_to_cummean(mean_mean[w_id,bc_id,0,5].reshape(1,chunks-1,1))
        c = scores_to_cummean(mean_dsca[w_id,bc_id,0,-1].reshape(1, chunks-1, 1))
        d = scores_to_cummean(mean_prev[w_id,bc_id,0,-1].reshape(1, chunks-1, 1))

        # ax.plot(gaussian_filter1d(raw_res[w_id], sig), label='base clf', color='orange')
        # ax.plot(gaussian_filter1d(mean_mean[w_id,bc_id,0,5],sig), ls='--', label='MEAN', c='tomato')
        # ax.plot(gaussian_filter1d(mean_dsca[w_id,bc_id,0,-1],sig), ls='--', label='DSCA', c='dodgerblue')

        ax.plot(a[0,:,0], label='base clf', color='black')
        ax.plot(b[0,:,0], ls='--', label='MEAN', c='dodgerblue')
        ax.plot(d[0,:,0], ls='--', label='PREV', c='orange')
        ax.plot(c[0,:,0], ls='--', label='DSCA', c='tomato')


        ax.set_ylim(.5,.9)
        ax.set_xlim(0, 500)
        ax.grid(ls=":")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()

fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)

plt.tight_layout()
fig.subplots_adjust(bottom=.12)
plt.savefig('foo.png')
plt.savefig('figures/e2.eps')
plt.savefig('figures/e2.png')


plt.clf()
fig, axx = plt.subplots(3,5,figsize=(10, 5), sharex=True, sharey=True)


for bc_id, bc in enumerate(base_clfs):
    axx[0,bc_id].set_title(base_clfs_names[bc_id])
    axx[2,bc_id].set_xlabel('chunk')

    #raw clf res
    raw_res = mean_raw_clfs_res[[2,5,8],bc_id]
    for w_id in range(3):
        ax = axx[w_id, bc_id]
        axx[w_id,0].set_ylabel(wn[w_id])

        a = (raw_res[w_id].reshape(1,chunks-1,1))
        b = (mean_mean[w_id,bc_id,0,5].reshape(1,chunks-1,1))
        c = (mean_dsca[w_id,bc_id,0,-1].reshape(1, chunks-1, 1))
        d = (mean_prev[w_id,bc_id,0,-1].reshape(1, chunks-1, 1))


        ax.plot(gaussian_filter1d(a[0,:,0], sig), label='base clf', color='black')
        ax.plot(gaussian_filter1d(b[0,:,0], sig), ls='--', label='MEAN', c='dodgerblue')
        ax.plot(gaussian_filter1d(d[0,:,0], sig), ls='--', label='PREV', c='orange')
        ax.plot(gaussian_filter1d(c[0,:,0], sig), ls='--', label='DSCA', c='tomato')


        ax.set_ylim(.5,1)
        ax.set_xlim(0, 500)
        ax.grid(ls=":")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        handles, labels = ax.get_legend_handles_labels()

fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)

# plt.legend()
plt.tight_layout()
fig.subplots_adjust(bottom=.17)
plt.savefig('foo.png')
plt.savefig('figures/e2_bac.eps')
plt.savefig('figures/e2_bac.png')
