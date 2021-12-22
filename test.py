import numpy as np
import strlearn as sl
from meta import Meta
from DSCA import DSCA
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

weights = (.5, .25)
# weights = (2, 5, 1.0)

y_flip = (.01, .01)


stream = sl.streams.StreamGenerator(
    weights=weights,
    random_state=1263,
    y_flip=y_flip,
    n_drifts=0,
    n_features=8,
    n_informative=8,
    n_redundant=0,
    n_repeated=0,
    n_clusters_per_class=1,
    n_chunks=200,
    chunk_size=200,
    class_sep=1,
)

meta = Meta(base_clf=GaussianNB(), prior_estimator=DSCA())
gnb = GaussianNB()

eval = sl.evaluators.TestThenTrain(verbose=True)
eval.process(stream, [gnb, meta])

fig, ax = plt.subplots(2, 2, figsize=(12,12))

ax[0,0].plot(meta.prior_estimator.calculated_priors_list, label = 'estimation')
ax[0,0].plot(meta.prior_estimator.errs, label = 'estim. error')
ax[0,0].set_title('dsca')
ax[0,0].legend()

ax[1,0].plot(eval.scores[0,:,0], label = 'acc')
ax[1,0].plot(eval.scores[0,:,1], label = 'bac')
ax[1,0].set_title('acc = %.3f, bac = %.3f' % (np.mean(eval.scores[0,:,0]), np.mean(eval.scores[0,:,1])))
ax[1,0].legend()


ax[0,1].plot(np.array(meta.prior_estimator.priors)[:,0], label = 'prior')
ax[0,1].set_title('gnb')
ax[0,1].legend()

ax[1,1].plot(eval.scores[1,:,0], label = 'acc')
ax[1,1].plot(eval.scores[1,:,1], label = 'bac')
ax[1,1].set_title('acc = %.3f, bac = %.3f' % (np.mean(eval.scores[1,:,0]), np.mean(eval.scores[1,:,1])))
ax[1,1].legend()


plt.tight_layout()
plt.savefig('foo.png')