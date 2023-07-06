import numpy as np

class LogisticRegression:
    def __init__(self):
        """
        :param
        :return
        """
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, learning_rate=0.01: float, num_iterations=1000: int, penalty=None: str):
        """
        
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.penalty = penalty
        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0

        # Perform gradient descent
        for _ in range(self.num_iterations):
            # Calculate current predictions
            linear_model = np.dot(X, self.coef_) + self.intercept_
            y_pred = 1/(1+np.exp(-linear_model))

            # Calculate gradients
            dw = (1 / len(X)) * np.dot(X.T, (y_pred - y))
            db = (1 / len(X)) * np.sum(y_pred - y)

            # Update parameters
            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db    
            
    def predict(self, X):
        # Calculate the raw model output
        linear_model = np.dot(X, self.coef_) + self.intercept_

        # Apply the sigmoid function
        y_pred = 1/(1+np.exp(-linear_model))

        # Convert probabilities to class labels
        y_pred_labels = [1 if elem >= 0.5 else 0 for elem in y_pred]

        return np.array(y_pred_labels)
    
    def predict_proba(self, X, *args, **kwargs):
        # Calculate the raw model output
        linear_model = np.dot(X, self.coef_) + self.intercept_

        # Apply the sigmoid function
        y_pred = 1/(1+np.exp(-linear_model))

        return y_pred