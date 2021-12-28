import numpy as np
import matplotlib.pyplot as plt
import config

weights = config.str_weights()
weigths_names = config.str_weights_names()
borders = config.borders()
criteria = config.criteria()
base_clfs = config.base_clfs()
base_clfs_names = config.base_clf_names()
reps = 10
chunks = 500

res = np.load('res_e1.npy')
raw_clfs_res = res[:,:,:3]
mean_raw_clfs_res = np.mean(raw_clfs_res, axis=3)
mean_raw_clfs_res = np.mean(mean_raw_clfs_res, axis=0)[:,:,0] # (streams x clfs)

meta_res = res[:,:,3:]
meta_res = meta_res.reshape((reps,len(weights),len(base_clfs),len(criteria),len(borders),2,chunks-1))
# (reps x streams x base_clfs x criteria x  borders x (mean, dsca) x chunks-1)

mean_meta_res = np.mean(meta_res, axis=-1)
mean_meta_res = np.mean(mean_meta_res, axis=0)
# (streams x base_clfs x criteria x borders x (mean, dsca))

mean_mean = mean_meta_res[:,:,:,:,0]
mean_dsca = mean_meta_res[:,:,:,:,1]
# (streams x base_clfs x criteria x borders)

for bc_id, bc in enumerate(base_clfs):

    # plot
    plt.clf()
    fig, axx = plt.subplots(3,3,figsize=(12,12),sharex=True)
    fig.suptitle("%s" % (base_clfs_names[bc_id]), fontsize=14)

    axx = axx.ravel()

    #raw clf res
    raw_res = mean_raw_clfs_res[:,bc_id]

    for w_id, w in enumerate(weights):
        ax = axx[w_id]

        ax.set_title(weigths_names[w_id])
        # ax.set_ylim(.5,1)
        ax.set_ylabel('BAC')
        ax.set_xlabel('border')
        ax.hlines(raw_res[w_id],0,borders[-1], label='base clf', color='orange')

        ax.plot(borders, mean_mean[w_id,bc_id,0,:], ls='--', label='MEAN c: %s' % criteria[0], c='tomato')
        ax.plot(borders, mean_mean[w_id,bc_id,1,:], ls=':', label='MEAN c: %s' % criteria[1], c='tomato')

        ax.plot(borders, mean_dsca[w_id,bc_id,0,:], ls='--', label='DSCA c: %s' % criteria[0], c='dodgerblue')
        ax.plot(borders, mean_dsca[w_id,bc_id,1,:], ls=':', label='DSCA c: %s' % criteria[1], c='dodgerblue')

    plt.legend()
    plt.tight_layout()
    fig.subplots_adjust(top=0.93)
    plt.savefig('figures/e1_%s.png' % base_clfs_names[bc_id])
    
    

