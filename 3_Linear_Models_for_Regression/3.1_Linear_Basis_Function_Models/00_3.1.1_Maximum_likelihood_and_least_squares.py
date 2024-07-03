# 00_3.1.1_Maximum_likelihood_and_least_squares

"""
Lecture: 3_Linear_Models_for_Regression/3.1_Linear_Basis_Function_Models
Content: 00_3.1.1_Maximum_likelihood_and_least_squares
"""

import numpy as np
from scipy.linalg import pinv

class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the linear regression model using maximum likelihood estimation.

        Parameters:
        X (np.ndarray): The input features, shape (N, M)
        y (np.ndarray): The target values, shape (N,)

        """
        # Adding a bias term (column of ones)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Compute the weights using the normal equation
        self.weights = pinv(X.T @ X) @ X.T @ y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted linear regression model.

        Parameters:
        X (np.ndarray): The input features, shape (N, M)

        Returns:
        np.ndarray: The predicted values, shape (N,)
        """
        # Adding a bias term (column of ones)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.weights

    def calculate_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate the residuals of the model.

        Parameters:
        X (np.ndarray): The input features, shape (N, M)
        y (np.ndarray): The target values, shape (N,)

        Returns:
        np.ndarray: The residuals, shape (N,)
        """
        predictions = self.predict(X)
        return y - predictions

# Example usage
if __name__ == "__main__":
    # Generating some example data
    np.random.seed(42)
    X = np.random.rand(100, 2)
    true_weights = np.array([2, 3, 5])  # Including bias term
    y = X @ true_weights[1:] + true_weights[0] + np.random.randn(100) * 0.5

    # Initialize and fit the model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)
    
    # Calculate residuals
    residuals = model.calculate_residuals(X, y)
    
    # Output the results
    print("Fitted weights:", model.weights)
    print("Predictions:", predictions[:5])
    print("Residuals:", residuals[:5])