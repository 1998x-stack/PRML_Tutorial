# 00_7.2.1_RVM_for_regression

"""
Lecture: 7_Sparse_Kernel_Machines/7.2_Relevance_Vector_Machines
Content: 00_7.2.1_RVM_for_regression
"""

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize
from typing import Tuple, List, Callable

class RelevanceVectorMachine:
    """
    相关向量机（Relevance Vector Machine, RVM）用于回归任务

    Attributes:
        kernel (Callable[[np.ndarray, np.ndarray], float]): 核函数
        alpha (np.ndarray): 超参数向量
        beta (float): 噪声精度
        weights (np.ndarray): 模型权重
        relevance_vectors (np.ndarray): 相关向量
        relevance_targets (np.ndarray): 相关向量对应的目标值
    """

    def __init__(self, kernel: Callable[[np.ndarray, np.ndarray], float] = None):
        self.kernel = kernel if kernel else self.linear_kernel
        self.alpha = None
        self.beta = None
        self.weights = None
        self.relevance_vectors = None
        self.relevance_targets = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        拟合模型，训练RVM

        Args:
            X (np.ndarray): 训练数据特征，形状为 (n_samples, n_features)
            y (np.ndarray): 训练数据标签，形状为 (n_samples,)
        """
        N = X.shape[0]
        self.alpha = np.ones(N)
        self.beta = 1.0

        # 计算核矩阵
        Phi = self.kernel_matrix(X, X)

        def objective(params: np.ndarray) -> float:
            """
            优化目标函数，负对数边际似然
            """
            alpha, beta = np.exp(params[:N]), np.exp(params[N])
            S_inv = np.diag(alpha) + beta * Phi
            L = cholesky(S_inv, lower=True)
            m = cho_solve((L, True), beta * y)

            log_likelihood = (
                0.5 * (N * np.log(2 * np.pi) - np.sum(np.log(alpha)) + N * np.log(beta))
                + 0.5 * y.T @ (beta * np.eye(N) - beta ** 2 * Phi @ cho_solve((L, True), Phi.T)) @ y
            )
            return log_likelihood

        params0 = np.log(np.hstack((self.alpha, self.beta)))
        result = minimize(objective, params0, method='L-BFGS-B')
        self.alpha, self.beta = np.exp(result.x[:N]), np.exp(result.x[N])

        S_inv = np.diag(self.alpha) + self.beta * Phi
        L = cholesky(S_inv, lower=True)
        self.weights = cho_solve((L, True), self.beta * y)
        self.relevance_vectors = X
        self.relevance_targets = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测新数据的目标值

        Args:
            X (np.ndarray): 测试数据特征，形状为 (n_samples, n_features)

        Returns:
            np.ndarray: 预测值，形状为 (n_samples,)
        """
        K = self.kernel_matrix(X, self.relevance_vectors)
        return K @ self.weights

    def kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        计算核矩阵

        Args:
            X1 (np.ndarray): 输入数据1
            X2 (np.ndarray): 输入数据2

        Returns:
            np.ndarray: 核矩阵
        """
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.kernel(X1[i], X2[j])
        return K

    @staticmethod
    def linear_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
        """
        线性核函数

        Args:
            x1 (np.ndarray): 输入向量1
            x2 (np.ndarray): 输入向量2

        Returns:
            float: 线性核的计算结果
        """
        return np.dot(x1, x2)

def main():
    # 示例数据
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [8, 9]
    ])
    y = np.array([2.3, 2.9, 3.8, 4.2, 5.1, 5.8, 6.9, 7.2])

    # 初始化和训练模型
    rvm = RelevanceVectorMachine()
    rvm.fit(X, y)

    # 测试数据
    X_test = np.array([
        [2, 3],
        [3, 4],
        [5, 6],
        [9, 10]
    ])

    # 预测
    predictions = rvm.predict(X_test)
    print("Predicted values:", predictions)

if __name__ == "__main__":
    main()