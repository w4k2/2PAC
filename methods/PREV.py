import numpy as np

"""
PREV prior probability estimator
"""

class PREV():
    def __init__(self):

        self.priors = []
        self.calculated_priors_list=[]
        self.errs=[]

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
                
    def estimate(self, X):
        self.chunk_estim_prior = self.priors[-1][0]

        # Save current chunk estimated pprior
        self.calculated_priors_list.append(self.chunk_estim_prior)
        
        return self.chunk_estim_prior
