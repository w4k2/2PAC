import numpy as np
import strlearn as sl
from methods.meta import Meta
from methods.DSCA import DSCA
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from scipy.signal import medfilt
from sklearn.base import clone

weights = (.025, .1)
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

base = GaussianNB()

meta_min = Meta(base_clf=clone(base), prior_estimator=DSCA(), correction=True, criterion='min', border=0.5)
meta_max = Meta(base_clf=clone(base), prior_estimator=DSCA(), correction=True, criterion='max', border=0.5)

eval = sl.evaluators.TestThenTrain(verbose=True, metrics=(accuracy_score, balanced_accuracy_score))
eval.process(stream, [meta_min, meta_max])

dif = eval.scores[0,:,0] != eval.scores[1,:,0]

print(eval.scores.shape)
print('min', eval.scores[0,:,0][dif])
print('max', eval.scores[1,:,0][dif])

plt.tight_layout()
plt.savefig('foo.png')
