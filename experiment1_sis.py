from PREV import PREV
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

borders = config.borders()
criteria = config.criteria()
base_clfs = config.base_clfs_no_mlp()

str_static = config.str_static()
str_weights = config.str_weights_sis()
n_chunks=str_static['n_chunks']

reps=10
random_states = np.random.randint(0,100000,reps)

pe_num = 3

meta_cnt = (len(base_clfs) * len(criteria) * len(borders) * pe_num ) + len(base_clfs)
results = np.zeros((reps, len(str_weights), meta_cnt, n_chunks-1, 1))
# reps x weights x (base clfs, criteria, border, prior estims) x chunks x BAC

estim_errs = np.zeros((reps, len(str_weights), meta_cnt, n_chunks-1))


t = reps*len(str_weights)
pbar = tqdm(total=t)

for r in range(reps):
    for w_id, w in enumerate(str_weights):

        # new instances of metas
        base_metas=[]
        base_metas.append(clone(base_clfs[0]))
        base_metas.append(clone(base_clfs[1]))
        base_metas.append(clone(base_clfs[2]))
        base_metas.append(clone(base_clfs[3]))
        # base_metas.append(clone(base_clfs[4]))
        for bc_id, bc in enumerate(base_clfs):
            for c_id, c in enumerate(criteria):
                for b_id, b in enumerate(borders):
                    base_metas.append(Meta(clone(bc), MEAN(), criterion=c, border=b))
                    base_metas.append(Meta(clone(bc), PREV(), criterion=c, border=b))
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

        for bm_i, bm in enumerate(base_metas[len(base_clfs):]):
            estim_errs[r, w_id, bm_i] = bm.prior_estimator.errs

        results[r, w_id] = eval.scores

np.save('res_e1_sis', results)

pbar.close()
