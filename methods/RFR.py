import numpy as np
from sklearn.ensemble import RandomForestRegressor

"""
Random Forest Regression prior probability estimator
"""

class RFR():
    def __init__(self, random_state=None):
        self.priors = []
        self.calculated_priors_list=[]
        self.errs=[]
        self.random_state = random_state
        self.reg = RandomForestRegressor(random_state=self.random_state)

    def feed(self, X, y, classes):
        self.X = np.copy(X)
        self.y = np.copy(y)
        self.classes = np.copy(classes)

        # Gather priors
        unique, counts  = np.unique(y, return_counts=True)
        if len(unique) == 1:
            if unique[0] == 0:
                self.priors.append([1., 0.])
            else:
                self.priors.append([0., 1.])
        else:
            self.priors.append(counts / y.shape)

        if hasattr(self, 'chunk_estim_prior'):
            self.errs.append(self.priors[-1][0] - self.chunk_estim_prior)
       
        X = np.arange(len(self.priors)).reshape(-1, 1)
        y = np.array(self.priors)[:,0].ravel()
        self.reg.fit(X, y)
                
    def estimate(self, X):

        self.chunk_estim_prior = self.reg.predict(np.array(len(self.priors)+1).reshape(-1, 1))[0]
        self.chunk_estim_prior = np.clip(self.chunk_estim_prior,0,1)

        # Save current chunk estimated pprior
        self.calculated_priors_list.append(self.chunk_estim_prior)
        
        return self.chunk_estim_prior
