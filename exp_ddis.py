from typing import OrderedDict
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

weights = config.str_weights_ddis3()
base_clfs = config.base_clfs()
str_static = config.str_static()
n_chunks=str_static['n_chunks']

reps=10
random_states = np.random.randint(0,100000,reps)

pe_num = 2

meta_cnt = (len(base_clfs) * pe_num ) + len(base_clfs)
results = np.zeros((reps, len(weights), meta_cnt, n_chunks-1, 1))
# reps x weights x (base clfs, prior estims) x chunks x BAC

estim_errs = np.zeros((reps, len(weights), len(base_clfs)*pe_num, n_chunks-1))


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
            base_metas.append(Meta(clone(bc), MEAN(), criterion='min', border=0.25))
            # base_metas.append(Meta(clone(bc), (random_state = 123), criterion=c, border=b))
            base_metas.append(Meta(clone(bc), DSCA(random_state = 123), criterion='min', border=0.5))

        # stream
        config = {
            **str_static,
            **weights[w],
            'random_state': random_states[r]
                    }
        stream = sl.streams.StreamGenerator(**config)

        # evaluate
        eval = sl.evaluators.TestThenTrain(verbose=True, metrics=(balanced_accuracy_score))
        eval.process(stream, base_metas)

        pbar.update(1)

        for bm_i, bm in enumerate(base_metas[3:]):
            estim_errs[r, w_id, bm_i] = bm.prior_estimator.calculated_priors_list

        results[r, w_id] = eval.scores

np.save('res_e_ddis', results)
np.save('estim_err_ddis', estim_errs)


pbar.close()



