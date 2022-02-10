from sklearn.base import BaseEstimator, ClassifierMixin, clone
import numpy as np
from imblearn.over_sampling import SMOTE

"""
border - od jakiego poziomu niezbalansowania korekcja predykcji (gdy estymacja a priori dla klasy
        mniejszosciowej jest mniejsza niz border, nastepuje korekcja)
"""

class Meta(BaseEstimator, ClassifierMixin):
    def __init__(self, base_clf, prior_estimator, criterion='min', correction=True, border=0.1):
        self.base_clf = base_clf
        self.prior_estimator = prior_estimator
        self.criterion=criterion
        self.correction = correction
        self.border = border

        self.clf = clone(base_clf)

    def partial_fit(self, X, y, classes):

        self.clf.partial_fit(X, y, classes)
        self.prior_estimator.feed(X, y, classes)
        return self

    def predict(self, X):

        estim_prior = self.prior_estimator.estimate(X)
        pred_proba = self.clf.predict_proba(X)

        # poza border
        if estim_prior > self.border and estim_prior < 1-self.border:
            return self.clf.predict(X)

        if estim_prior > 0.5: #wiekszosc to 0

            if self.correction:
                pred = self.clf.predict(X)
            else:
                pred = np.zeros((X.shape[0]))

            positive_class_samples = int(np.rint((1-estim_prior)*X.shape[0]))
            if positive_class_samples != 0:
                max_supp_1 = np.argsort(pred_proba[:,1])[-positive_class_samples:]
                min_supp_0 = np.argsort(pred_proba[:,0])[:positive_class_samples]

                if self.criterion == 'min':
                    pred[min_supp_0] = 1
                elif self.criterion == 'max':
                    pred[max_supp_1] = 1
                else:
                    exit()

        else: #wiekszosc to 1
            if self.correction:
                pred = self.clf.predict(X)
            else:
                pred = np.ones((X.shape[0]))

            negative_class_samples = int(np.rint(estim_prior*X.shape[0]))
            if negative_class_samples != 0:
              
                max_supp_0 = np.argsort(pred_proba[:,0])[-negative_class_samples:]
                min_supp_1 = np.argsort(pred_proba[:,1])[:negative_class_samples]

                if self.criterion == 'min':
                    pred[min_supp_1] = 0
                elif self.criterion == 'max':
                    pred[max_supp_0] = 0
                else:
                    exit()

        return pred
