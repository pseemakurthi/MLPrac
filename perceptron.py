imprt numpy as np
class Perceptron:
    """
        Params:
        eta : float
        eta is the learning rate

        n_iter(epoch) : int
        n_iter is the no of epoch before quitting the algorithms
    """

    def __init__(self, eta = 0.1, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter

    def net_input(self, X):
        """ calculates the net input """
        return np.dot(X,self.w_[1:]) + self.w_[0]

    def predict(self, Xi):
        """Returns 1 or -1 based on the net_input  """
        return np.where(self.net_input(X) > 0.0, 1 -1)


    def fit(X,y):
        """
        Input:
            X  feature matrix
            y target vector

        output:
            object
        """

        self.w_ = np.zeros(1 + X.shape[1]) #first is initialized to 1
        self.errors_ = []
        for _ in rage(self.n_iter):
            error = 0
            for target, xi in zip(X,y):
                update = self.eta * (target - np.predict(xi))
                self.w_[1:] = update * xi
                self.w_[0:] = update
                error += int(update != 0.0)
                self.errors_.append(error)
            return self
