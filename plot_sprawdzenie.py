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
errs = np.mean(np.load('results/errs_e1_all.npy'),axis=0)

raw_clfs_res = res[:,:,:len(base_clfs)]
mean_raw_clfs_res = np.mean(raw_clfs_res, axis=0)[:,:,:,0] # (streams x clfs x chunks)

meta_res = res[:,:,len(base_clfs):]
meta_res = meta_res.reshape((reps,len(weights),len(base_clfs),len(criteria),len(borders),pe,chunks-1))

errs = errs.reshape((len(weights), len(base_clfs), len(criteria), len(borders),pe,chunks-1))
# print(errs.shape)
# exit()
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
    fig, axx = plt.subplots(1,3,figsize=(12,5),sharex=True, sharey=True)
    fig.suptitle("%s" % (base_clfs_names[bc_id]), fontsize=14)

    axx = axx.ravel()

    #raw clf res
    raw_res = mean_raw_clfs_res[:,bc_id]

    for w_id, w in enumerate(config.str_weights_cdis()):

        ax = axx[w_id]

        ax.set_title(weigths_names[w_id])
        # ax.set_ylim(.5,1)
        ax.set_ylabel('BAC')
        ax.set_xlabel('chunk')

        a = (raw_res[3+w_id].reshape(1,chunks-1,1))
        b1 = (mean_mean[3+w_id,bc_id,0,5].reshape(1,chunks-1,1))
        b2 = (mean_mean[3+w_id,bc_id,0,-1].reshape(1,chunks-1,1))
      
        ax.plot(a[0,:,0], label='base clf', color='orange')
        ax.plot(b1[0,:,0], ls='--', label='MEAN 0.3', c='tomato')
        ax.plot(b2[0,:,0], ls='--', label='MEAN 0.5', c='blue')
        
        ax.plot(errs[3+w_id, bc_id,0,0,0], label='err mean', c='purple')

        ax.set_xlim(0, 500)
        ax.grid(ls=":")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.legend()
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig('foo.png')
    plt.savefig('figures/spr_%s.png' % base_clfs_names[bc_id])
    # plt.savefig('figures/e2_%s.png' % base_clfs_names[bc_id])
