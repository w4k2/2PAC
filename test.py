import numpy as np
import strlearn as sl
from meta import Meta
from DSCA import DSCA
from MEAN import MEAN
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from scipy.signal import medfilt
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
# from SEA2 import SEA2
from sklearn.linear_model import SGDClassifier
from RFR import RFR

# weights = (.015, .05)
weights = (4, 5, .75)
# weights = [0.025, 0.975]
# weights=[0.975, 0.025]
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
    n_clusters_per_class=2,
    n_chunks=500,
    chunk_size=200,
    class_sep=1,
)

# base = SEA2(KNeighborsClassifier())
# base = SGDClassifier(loss='modified_huber')
# base = MLPClassifier()
base = GaussianNB()

# meta = Meta(base_clf=clone(base), prior_estimator=DSCA(), correction=True, criterion='min', resample=False, border=0.1)
# meta = Meta(base_clf=clone(base), prior_estimator=MEAN(), correction=True, criterion='min')
meta = Meta(base_clf=clone(base), prior_estimator=RFR(), correction=True, criterion='min', border=0.01)
gnb = clone(base)

eval = sl.evaluators.TestThenTrain(verbose=True, metrics=(accuracy_score, balanced_accuracy_score))
eval.process(stream, [meta, gnb])

print(eval.scores.shape)

fig, ax = plt.subplots(2, 2, figsize=(12,12))
kernel=11

ax[0,0].plot(meta.prior_estimator.calculated_priors_list, label = 'estimation')
ax[0,0].plot(meta.prior_estimator.errs, label = 'estim. error')
ax[0,0].set_title('dsca')
ax[0,0].set_ylim(0,1)
ax[0,0].legend()

ax[1,0].plot(medfilt(eval.scores[0,:,0], kernel), label = 'acc')
ax[1,0].plot(medfilt(eval.scores[0,:,1], kernel), label = 'bac')
ax[1,0].set_ylim(0.5,1)
ax[1,0].set_title('acc = %.3f, bac = %.3f' % (np.mean(eval.scores[0,:,0]), np.mean(eval.scores[0,:,1])))
ax[1,0].legend()


ax[0,1].plot(np.array(meta.prior_estimator.priors)[:,0], label = 'prior')
ax[0,1].set_title('gnb')
ax[0,1].set_ylim(0,1)
ax[0,1].legend()

ax[1,1].plot(medfilt(eval.scores[1,:,0], kernel), label = 'acc')
ax[1,1].plot(medfilt(eval.scores[1,:,1], kernel), label = 'bac')
ax[1,1].set_ylim(0.5,1)
ax[1,1].set_title('acc = %.3f, bac = %.3f' % (np.mean(eval.scores[1,:,0]), np.mean(eval.scores[1,:,1])))
ax[1,1].legend()


plt.tight_layout()
plt.savefig('foo.png')