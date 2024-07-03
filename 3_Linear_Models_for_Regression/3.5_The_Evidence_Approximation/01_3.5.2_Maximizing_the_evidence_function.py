# 01_3.5.2_Maximizing_the_evidence_function

"""
Lecture: 3_Linear_Models_for_Regression/3.5_The_Evidence_Approximation
Content: 01_3.5.2_Maximizing_the_evidence_function
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

    def update_hyperparameters(self, X: np.ndarray, t: np.ndarray):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        eigvals = np.linalg.eigvalsh(self.beta * X.T @ X)
        gamma = np.sum(eigvals / (self.alpha + eigvals))
        
        self.alpha = gamma / np.sum(self.m_N**2)
        self.beta = (X.shape[0] - gamma) / np.sum((t - X @ self.m_N)**2)

    def evidence(self, X: np.ndarray, t: np.ndarray) -> float:
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        A = self.alpha * np.eye(X.shape[1]) + self.beta * X.T @ X
        E_mN = (self.beta / 2) * np.sum((t - X @ self.m_N)**2) + (self.alpha / 2) * np.sum(self.m_N**2)
        log_evidence = (X.shape[1] / 2) * np.log(self.alpha) + (X.shape[0] / 2) * np.log(self.beta) - E_mN - (1 / 2) * np.log(np.linalg.det(A)) - (X.shape[0] / 2) * np.log(2 * np.pi)
        return log_evidence

if __name__ == "__main__":
    X_train = np.array([[0.1], [0.4], [0.7], [1.0]])
    t_train = np.array([1.1, 1.9, 3.0, 4.2])
    
    model = BayesianLinearRegression(alpha=1.0, beta=25.0)
    model.fit(X_train, t_train)
    evidence = model.evidence(X_train, t_train)
    print("初始证据函数对数值: ", evidence)
    
    model.update_hyperparameters(X_train, t_train)
    print("更新后超参数 α: ", model.alpha)
    print("更新后超参数 β: ", model.beta)
    
    evidence = model.evidence(X_train, t_train)
    print("更新后证据函数对数值: ", evidence)
