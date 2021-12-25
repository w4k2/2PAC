import numpy as np
import strlearn as sl
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier


def str_static():
    return {
        'n_chunks': 500,
        'n_features': 8,
        'n_informative': 8,
        'n_redundant': 0,
        'y_flip': (.01, .01)
    }

def str_weights():
    return {
    'sis_10': {'weights':[0.1, 0.9]},
    'sis_5': {'weights':[0.05, 0.95]},
    'sis_2,5': {'weights':[0.025, 0.975]},    
    'cdis_75': {'weights':(4, 5, .75)},
    'cdis_90': {'weights':(4, 5, .9)},
    'cdis_100': {'weights':(4, 5, 1.)},    
    'ddis_10': {'weights':(.1, .05)},
    'ddis_5': {'weights':(.05, .05)},
    'ddis_2,5': {'weights':(.025, .05)}
    }

def criteria():
    return ['min', 'max']

def corrections():
    return [False, True] # chyba zawsze true dla ulatwienia

def borders():
    #tylko dla correction == True
    return np.linspace(0.01, 0.5, 10)

def base_clfs():
    return [
        GaussianNB(),
        MLPClassifier(),
        SGDClassifier(loss='modified_huber'),
    ]
