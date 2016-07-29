from sklearn.metrics import accuracy_score
from sklearn.base import base
from sklearn.preprocessing import train_test_split
from itertool import combinations
import numpy as np

class SBS:
    '''
        This class simulates the sequential backward selection similar to stepAIC from R
    '''
    def __init__(self, estimator, k_features, scoring = accuracy_score, test_size = 0.2, random_state = 42):
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.scoring = scoring
        self.random_state = random_state
        self.test_size = test_size

    def _cal_score(self, X_train, X_test, y_train, y_test, indices):
        '''
            Takes the list of indices and Calculates the scores with selected indices
        '''

        self.estimator.fit(X_train[:, indices], y_train)
        y_predict = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_predict)
        return score

    def transform(self, X):
        return X[:, self.indices_]

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = self.test_size, random_state = self.random_state)

        dim = X.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._cal_score(X_train, X_test, y_train, y_test,self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:

            scores = []
            subsets = []

            for p in combinations(self.indices_, r = dim -1):
                score = self._cal_score(X_train, X_test, y_train, y_test,p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        return self
