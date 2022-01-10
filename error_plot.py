import numpy as np
import matplotlib.pyplot as plt
import config
from scipy.ndimage import gaussian_filter1d

weights = config.str_weights()
weigths_names = config.str_weights_names()
borders = config.borders()
criteria = config.criteria()
base_clfs = config.base_clfs()
base_clfs_names = config.base_clf_names()
reps = 10
chunks = 500
pe=3

np.random.seed(1231)
random_states = np.random.randint(0,100000,reps)

s=3

errs = np.load('results/errs_e1_all.npy')
errs_mean = np.mean(np.abs(errs[:,:,0,:]), axis=0)
errs_prev = np.mean(np.abs(errs[:,:,1,:]), axis=0)
errs_dsca = np.mean(np.abs(errs[:,:,2,:]), axis=0)

print(errs.shape)

fig, axx = plt.subplots(3,3,figsize=(7*1.618, 7),
                        sharex=True)
axx = axx.ravel()

xx = np.linspace(1,500,499)

for w_id, w in enumerate(weights):
    ax = axx[w_id]
    ax.set_title(weigths_names[w_id])
    ax.set_xlabel('chunk id')

    level=np.zeros((499))

    c = gaussian_filter1d(errs_mean[w_id],s)
    b = gaussian_filter1d(errs_prev[w_id],s)
    a = gaussian_filter1d(errs_dsca[w_id],s)

    ax.fill_between(xx, level, level+a, color='tomato', label='DSCA')
    ax.fill_between(xx, level+a, level+a+b, color='dodgerblue', label='PREV')
    ax.fill_between(xx, level+a+b, level+a+b+c, color='orange', label='MEAN')

    ax.grid(ls=":")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0,500)

for ax in axx[:3]:
   ax.set_ylim(0,0.11)
for ax in axx[3:6]:
    ax.set_ylim(0,0.7)
for ax in axx[6:]:
    ax.set_ylim(0,0.3)

for ax in axx[::3]:
    ax.set_ylabel('estimation error')

plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('figures/errors.png')
plt.savefig('figures/errors.eps')
plt.savefig('foo.png')
