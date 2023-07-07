import numpy as np

def sigmoid(z: np.ndarray):
    """
    :param z is a regression to be made into a sigmoid    

    """
    return 1/(1 + np.exp(-z))


class LogisticRegression(X, y):
    """
    """

    def __init__(self, X, y):
        """

        """
        self.coef_ = coef_
        self.intercept_ = intercept

    def fit(self, X, y, penalty="l1", learning_rate = 0.1, num_iter = 1000):
        """
        
        """
        
        self.coef_ = np.ndarray()
        self.intercept = 0 


    def predict(self, X, y):
        """
        
        """
        pass
    def predictProba(self, X, y):
        """
        
        """
        pass

