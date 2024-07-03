# 05_1.2.6_Bayesian_curve_fitting

"""
Lecture: 1_Introduction/1.2_Probability_Theory
Content: 05_1.2.6_Bayesian_curve_fitting
"""

import numpy as np
from scipy.linalg import solve

class BayesianCurveFitting:
    """
    贝叶斯曲线拟合类
    
    该类实现贝叶斯方法对曲线拟合的处理，包括参数的后验分布和预测分布的计算。
    
    Attributes:
        degree (int): 多项式的阶数
        alpha (float): 先验分布的精度参数
        beta (float): 噪声分布的精度参数
    """

    def __init__(self, degree: int, alpha: float, beta: float):
        """
        初始化贝叶斯曲线拟合类

        Args:
            degree (int): 多项式的阶数
            alpha (float): 先验分布的精度参数
            beta (float): 噪声分布的精度参数
        """
        self.degree = degree
        self.alpha = alpha
        self.beta = beta
        self.coefficients_mean = None
        self.coefficients_cov = None

    def fit(self, x: np.ndarray, t: np.ndarray) -> None:
        """
        拟合贝叶斯曲线

        Args:
            x (np.ndarray): 输入数据
            t (np.ndarray): 目标值
        """
        assert x.shape[0] == t.shape[0], "输入数据和目标值的大小不匹配"
        assert x.ndim == 1, "输入数据应为一维数组"

        X = self._design_matrix(x)
        S_inv = self.alpha * np.eye(X.shape[1]) + self.beta * X.T @ X
        S = np.linalg.inv(S_inv)
        self.coefficients_mean = self.beta * S @ X.T @ t
        self.coefficients_cov = S

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        使用拟合的贝叶斯曲线进行预测

        Args:
            x (np.ndarray): 输入数据

        Returns:
            np.ndarray: 预测值
        """
        assert self.coefficients_mean is not None, "请先拟合模型"
        X = self._design_matrix(x)
        mean = X @ self.coefficients_mean
        variance = 1 / self.beta + np.sum(X @ self.coefficients_cov * X, axis=1)
        return mean, variance

    def _design_matrix(self, x: np.ndarray) -> np.ndarray:
        """
        构建设计矩阵

        Args:
            x (np.ndarray): 输入数据

        Returns:
            np.ndarray: 设计矩阵
        """
        return np.vander(x, self.degree + 1, increasing=True)

    def calculate_rmse(self, x: np.ndarray, t: np.ndarray) -> float:
        """
        计算均方根误差（RMSE）

        Args:
            x (np.ndarray): 输入数据
            t (np.ndarray): 目标值

        Returns:
            float: 均方根误差
        """
        mean, _ = self.predict(x)
        return np.sqrt(np.mean((mean - t) ** 2))

# 示例数据
x_train = np.array([0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 3.1])
t_train = np.sin(x_train) + 0.1 * np.random.randn(len(x_train))

# 创建贝叶斯曲线拟合实例
degree = 3
alpha = 2.0
beta = 25.0
bayesian_curve_fitting = BayesianCurveFitting(degree, alpha, beta)

# 拟合模型
bayesian_curve_fitting.fit(x_train, t_train)

# 进行预测
x_test = np.linspace(0, 3.5, 100)
mean, variance = bayesian_curve_fitting.predict(x_test)

# 打印均方根误差
rmse = bayesian_curve_fitting.calculate_rmse(x_train, t_train)
print(f"均方根误差: {rmse:.4f}")

# 打印预测结果的均值和方差
print("预测结果的均值:", mean)
print("预测结果的方差:", variance)
