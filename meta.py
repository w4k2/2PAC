from sklearn.base import BaseEstimator, ClassifierMixin, clone
import numpy as np

"""
criterion:
'min' - minimalne wsparcie dla lasy wiekszosciowej
'max' - maksymalne wparcie dla klasy mniejszosciowej

correction - czy uzywac predykcji klayfikatora bazowego jako bazy
"""

class Meta(BaseEstimator, ClassifierMixin):
    def __init__(self, base_clf, prior_estimator, criterion='min', correction = False):
        self.base_clf = base_clf
        self.prior_estimator = prior_estimator
        self.criterion=criterion
        self.correction = correction

        self.clf = clone(base_clf)

    def partial_fit(self, X, y, classes):
        self.clf.partial_fit(X, y, classes)
        self.prior_estimator.feed(X, y, classes)
        return self

    def predict(self, X):

        if self.correction:
            pred = self.clf.predict(X)
        else:
            pred = np.ones((X.shape[0]))

        estim_prior = self.prior_estimator.estimate(X)
        negative_class_samples = int(np.rint(estim_prior*X.shape[0]))
        
        pred_proba = self.clf.predict_proba(X)

        max_supp_0 = np.argsort(pred_proba[:,0])[-negative_class_samples:]
        min_supp_1 = np.argsort(pred_proba[:,1])[:negative_class_samples]

        if self.criterion == 'min':
            pred[min_supp_1] = 0
        elif self.criterion == 'max':
            pred[max_supp_0] = 0
        else:
            exit()

        return pred
