import numpy as np
import strlearn as sl
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from SEA2 import SEA2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from skmultiflow.trees import HoeffdingTreeClassifier


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
    'ddis_10': {'weights':(.1, .1)},
    'ddis_5': {'weights':(.05, .1)},
    'ddis_2,5': {'weights':(.025, .1)}
    }

def str_weights_names():
    return ['SIS .1', 'SIS .05', 'SIS .025', 'CDIS .75', 'CDIS 0.9', 'CDIS 1.', 'DDIS .1/.1','DDIS .05/.1','DDIS .025/.1']

def str_weights_ddis():
    return {   
    'ddis_10_2,5': {'weights':(.1, .025)},
    'ddis_5_2,5': {'weights':(.05, .025)},
    'ddis_2,5_2,5': {'weights':(.025, .025)},
    'ddis_1_2,5': {'weights':(.01, .025)},
    'ddis_10_5': {'weights':(.1, .05)},
    'ddis_5_5': {'weights':(.05, .05)},
    'ddis_2,5_5': {'weights':(.025, .05)},
    'ddis_1_5': {'weights':(.01, .05)},
    'ddis_10_10': {'weights':(.1, .1)},
    'ddis_5_10': {'weights':(.05, .1)},
    'ddis_2,5_10': {'weights':(.025, .1)},
    'ddis_1_10': {'weights':(.01, .1)}
    }
    
def str_weights_names_ddis():
    return [
        'DDIS .1/.025','DDIS .05/.025','DDIS .025/.025', 'DDIS .01/.025',
        'DDIS .1/.05','DDIS .05/.05','DDIS .025/.05', 'DDIS .01/.05',
        'DDIS .1/.1','DDIS .05/.1','DDIS .025/.1', 'DDIS .01/.1']


def criteria():
    return ['min', 'max']

# def corrections():
#     return [False, True] # chyba zawsze true dla ulatwienia ??

def borders():
    #tylko dla correction == True
    return np.linspace(0.01, 0.5, 10)

def base_clfs():
    return [
        GaussianNB(),
        MLPClassifier(),
        # # SGDClassifier(loss='modified_huber'),
        SEA2(KNeighborsClassifier()),
        SEA2(SVC(probability=True)),
        # HoeffdingTreeClassifier()
    ]

def base_clf_names():
    return ['GNB', 'MLP', 'KNN', 'SVM']
