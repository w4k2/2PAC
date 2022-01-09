import numpy as np
import matplotlib.pyplot as plt
import config
import strlearn as sl
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

str_static = config.str_static()

priors=[]

for r in random_states:

    for w_id, w in enumerate(weights):
        config = {
            **str_static,
            **weights[w],
            'random_state': random_states[0]
                    }
        stream = sl.streams.StreamGenerator(**config)

        
        s_priors = []
        for i in range(500):
            unique, counts  = np.unique(stream.get_chunk()[1], return_counts=True)
            if len(unique) == 1:
                if unique[0] == 0:
                    s_priors.append(1.)
                else:
                    s_priors.append(0.)
            else:
                s_priors.append(counts[0] / 200)
            
        priors.append(s_priors)

priors = np.array(priors).reshape(reps,9,500)
priors = priors[:,:,1:]

print(priors.shape)

mean_priors = np.mean(priors, axis=0)

fig, axx = plt.subplots(3,1,figsize=(6, 6*1.618), sharex=True, sharey=True)
axx = axx.ravel()

t = ['SIS', 'CDIS', 'DDIS']
cols=['dodgerblue', 'orange', 'tomato']
c = 0
for i in range(3):
    ax = axx[i]

    ax.set_title(t[i])
    ax.set_ylabel('prior probability')
    ax.set_xlabel('chunk id')
    ax.set_ylim(0,1)

    for j in range(3):
        ax.plot(gaussian_filter1d(mean_priors[c],1), color=cols[c%3], label=weigths_names[c])
        c+=1
    
    ax.legend()

plt.tight_layout()
plt.savefig('figures/priors.png')
plt.savefig('foo.png')
