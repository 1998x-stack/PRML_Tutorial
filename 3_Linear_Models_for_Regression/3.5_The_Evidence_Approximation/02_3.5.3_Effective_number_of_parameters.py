# 02_3.5.3_Effective_number_of_parameters

"""
Lecture: 3_Linear_Models_for_Regression/3.5_The_Evidence_Approximation
Content: 02_3.5.3_Effective_number_of_parameters
"""

import numpy as np
from scipy.linalg import inv

class BayesianLinearRegression:
    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta
        self.m_N = None
        self.S_N = None

    def fit(self, X: np.ndarray, t: np.ndarray):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        S_0_inv = self.alpha * np.eye(X.shape[1])
        self.S_N = inv(S_0_inv + self.beta * X.T @ X)
        self.m_N = self.beta * self.S_N @ X.T @ t

    def effective_number_of_parameters(self, X: np.ndarray) -> float:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        eigvals = np.linalg.eigvalsh(self.beta * X.T @ X)
        gamma = np.sum(eigvals / (self.alpha + eigvals))
        return gamma

if __name__ == "__main__":
    X_train = np.array([[0.1], [0.4], [0.7], [1.0]])
    t_train = np.array([1.1, 1.9, 3.0, 4.2])
    
    model = BayesianLinearRegression(alpha=1.0, beta=25.0)
    model.fit(X_train, t_train)
    gamma = model.effective_number_of_parameters(X_train)
    
    print("有效参数数目 γ: ", gamma)
