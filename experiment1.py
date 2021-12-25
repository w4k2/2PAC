"""
Hyperparameter optimization

base_clf: [GNB, MLP, SDG(log/modified_huber)] -- optymalizacja dla kazdego

prior_estimator: [DSCA, RFR, MEAN]

parametry:
criterion: [min, max]
correction (raczej tylko true)
resample (raczej nie?)
border [0.01,0.5]

strumienie:

2 clusters per chunk
500 chunks x 200 samples
features: 8

weights:
- SIS
[0.1, 0.9]
[0.05, 0.95]
[0.025, 0.975]
- CDIS
(4, 5, .75)
(4, 5, .9)
(4, 5, 1.)
- DDIS
(.1, .05)
(.05, .05)
(.025, .05)
"""

import config
import strlearn as sl
import numpy as np
from meta import Meta
from sklearn.base import clone
from MEAN import MEAN
from RFR import RFR
from DSCA import DSCA
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm


np.random.seed(1231)

weights = config.str_weights()
borders = config.borders()
criteria = config.criteria()
base_clfs = config.base_clfs()

str_static = config.str_static()
str_weights = config.str_weights()
n_chunks=str_static['n_chunks']

reps=10
random_states = np.random.randint(0,100000,reps)

pe_num = 2

meta_cnt = (len(base_clfs) * len(criteria) * len(borders) * pe_num ) + 3
results = np.zeros((reps, len(weights), meta_cnt, n_chunks-1, 1))
# reps x weights x (base clfs, criteria, border, prior estims) x chunks x BAC

t = reps*len(weights)
pbar = tqdm(total=t)

for r in range(reps):
    for w_id, w in enumerate(weights):

        # new instances of metas
        base_metas=[]
        base_metas.append(clone(base_clfs[0]))
        base_metas.append(clone(base_clfs[1]))
        base_metas.append(clone(base_clfs[2]))
        for bc_id, bc in enumerate(base_clfs):
            for c_id, c in enumerate(criteria):
                for b_id, b in enumerate(borders):
                    base_metas.append(Meta(clone(bc), MEAN(), criterion=c, border=b))
                    # base_metas.append(Meta(clone(bc), (random_state = 123), criterion=c, border=b))
                    base_metas.append(Meta(clone(bc), DSCA(random_state = 123), criterion=c, border=b))
        
        # stream
        config = {
            **str_static,
            **str_weights[w],
            'random_state': random_states[r]
                    }
        stream = sl.streams.StreamGenerator(**config)

        # evaluate
        eval = sl.evaluators.TestThenTrain(verbose=True, metrics=(balanced_accuracy_score))
        eval.process(stream, base_metas)

        pbar.update(1)

        results[r, w_id] = eval.scores

np.save('res_e1', results)

pbar.close()



