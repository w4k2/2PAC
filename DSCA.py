import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone

"""
DSCA - Dynamic Statictical Concept Analysis

Params:
window - window size for dsca
max_iter - max number of epochs for dsca
base_reg - regressor for dsca

self.calculated_priors_list - contains results or prior proba estimation
"""

class DSCA(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                window=1,
                max_iter=250,
                base_reg=MLPRegressor(random_state=1410),
                random_state=None):

        self.random_state = random_state
        self.window = window
        self.max_iter = max_iter

        # Initializer
        self.priors = []
        self.calculated_priors_list=[]
        self.dsca_X = []
        self.dsca_reg_0 = clone(base_reg)
        self.dsca_reg_1 = clone(base_reg)
        self.errs=[]

        np.random.seed(self.random_state)


    def feed(self, X, y, classes):
        self.X = np.copy(X)
        self.y = np.copy(y)
        self.classes_ = np.copy(classes)

        # Gather priors
        _ = np.unique(y, return_counts=True)
        self.priors.append(_[1][self.classes_[_[0]]] / y.shape)
        apriors = np.array(self.priors)

        # Dynamic statistical concept analysis
        self.dsca_X.append(np.concatenate((np.mean(self.X, axis=0), np.std(self.X, axis=0))))

        self.chunk_real_prior = apriors[-1,0]

        if len(apriors) > 1:
            err = np.abs(self.chunk_real_prior - self.chunk_estim_prior)
            self.errs.append(err)
            n_iter = 1 + int(self.max_iter * err)
        else:
            n_iter = self.max_iter
            

        window = self.window if self.window > len(apriors) else len(apriors)
        for i in range(n_iter):
            self.dsca_reg_0.partial_fit(self.dsca_X[-window:], np.array(apriors[-window:,0]))
            self.dsca_reg_1.partial_fit(self.dsca_X[-window:], np.array(apriors[-window:,1]))


        
    def estimate(self, X):
        # Get dsca priori for current data
        dsca_x = np.concatenate((np.mean(X, axis=0), np.std(X, axis=0)))
        dsca_y_pred_0 = self.dsca_reg_0.predict([dsca_x])[0]
        dsca_y_pred_1 = self.dsca_reg_1.predict([dsca_x])[0]
        self.chunk_estim_prior = dsca_y_pred_0 / (dsca_y_pred_0 + dsca_y_pred_1)
        self.chunk_estim_prior = 1 if self.chunk_estim_prior > 1 else 0 if self.chunk_estim_prior < 0 else self.chunk_estim_prior #clip?

        # Save current chunk estimated pprior
        self.calculated_priors_list.append(self.chunk_estim_prior)
        
        return self.chunk_estim_prior
