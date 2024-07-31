import numpy as np
import matplotlib.pyplot as plt
import config

weights = config.str_weights()
weigths_names = config.str_weights_names()
borders = config.borders()
criteria = config.criteria()
# base_clfs = config.base_clfs()
base_clfs_names = config.base_clf_names()
reps = 10
chunks = 500

pe=3

res = np.load('results/res_e1_all.npy')
raw_clfs_res = res[:,:,:len(base_clfs_names)]
print(res.shape) # reps, streams, base_clfs, chunks
print(base_clfs_names)
print(raw_clfs_res.shape)

mean_raw_clfs_res = np.mean(raw_clfs_res, axis=3)
mean_raw_clfs_res = np.mean(mean_raw_clfs_res, axis=0)[:,:,0] # (streams x clfs)

meta_res = res[:,:,len(base_clfs_names):]
print(meta_res.shape)


meta_res = meta_res.reshape((reps,len(weights),len(base_clfs_names),len(criteria),len(borders),pe,chunks-1))
print(criteria,borders)
# (reps x streams x base_clfs x criteria x  thresholds x (mean, prev, dsca) x chunks-1)

meta_res = meta_res[:,:,:,0]
# (reps x streams x base_clfs x thresholds x (mean, prev, dsca) x chunks-1)

print(meta_res.shape)

mean_meta_res = np.mean(meta_res, axis=-1)
mean_meta_res = np.mean(mean_meta_res, axis=0)
# (streams x base_clfs x  thresholds x (mean, prev, dsca))

aa = np.copy(mean_meta_res[:,:,:,0]) # SWAP MEAN AND PREV
mean_meta_res[:,:,:,0] = np.copy(mean_meta_res[:,:,:,1])
mean_meta_res[:,:,:,1] = aa

print(mean_meta_res.shape)

for clf_id in range(5):
       
    fig, ax = plt.subplots(3,3,figsize=(10,6), sharex=True)
    ax = ax.ravel()
    plt.suptitle('base classifier: %s' % base_clfs_names[clf_id])

    for i in range(9):
        ax[i].set_title(weigths_names[i])
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].grid(ls=':')
        
        if i%3==0:
            ax[i].set_ylabel('balanced accuracy')
        if i>5:
            ax[i].set_xlabel('threshold')
            
        for prior_id in range(3):
            ax[i].plot(borders, mean_meta_res[i,clf_id, :, prior_id], label=['PREV', 'MEAN', 'DSCA'][prior_id], color=['r', 'g', 'b'][prior_id])
            
    ax[0].legend(frameon=False)
    plt.tight_layout()
    plt.savefig('foo.png')
    plt.savefig('figures/e1_v2_%s.png' % base_clfs_names[clf_id])
    plt.savefig('figures/e1_v2_%s.eps' % base_clfs_names[clf_id])


    
