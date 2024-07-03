# 04_1.2.5_Curve_fitting_re-visited

"""
Lecture: 1_Introduction/1.2_Probability_Theory
Content: 04_1.2.5_Curve_fitting_re-visited
"""

import numpy as np
import scipy.linalg

class PolynomialCurveFitting:
    """
    多项式曲线拟合类

    该类使用最小二乘法和正则化技术进行多项式曲线拟合。

    Attributes:
        degree (int): 多项式的阶数
        regularization_param (float): 正则化参数
    """

    def __init__(self, degree: int, regularization_param: float = 0.0):
        """
        初始化多项式曲线拟合类

        Args:
            degree (int): 多项式的阶数
            regularization_param (float): 正则化参数
        """
        self.degree = degree
        self.regularization_param = regularization_param
        self.coefficients = None

    def fit(self, x: np.ndarray, t: np.ndarray) -> None:
        """
        拟合多项式曲线

        Args:
            x (np.ndarray): 输入数据
            t (np.ndarray): 目标值
        """
        assert x.shape[0] == t.shape[0], "输入数据和目标值的大小不匹配"
        assert x.ndim == 1, "输入数据应为一维数组"
        
        X = self._design_matrix(x)
        if self.regularization_param > 0:
            I = np.eye(X.shape[1])
            self.coefficients = np.linalg.solve(
                X.T @ X + self.regularization_param * I, X.T @ t
            )
        else:
            self.coefficients = np.linalg.solve(X.T @ X, X.T @ t)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        使用拟合的多项式进行预测

        Args:
            x (np.ndarray): 输入数据

        Returns:
            np.ndarray: 预测值
        """
        assert self.coefficients is not None, "请先拟合模型"
        X = self._design_matrix(x)
        return X @ self.coefficients

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
        predictions = self.predict(x)
        return np.sqrt(np.mean((predictions - t) ** 2))

# 示例数据
x = np.array([0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9, 2.2, 2.5, 2.8, 3.1])
t = np.array([1.3, 1.5, 1.7, 2.0, 2.2, 2.4, 2.6, 2.9, 3.1, 3.3, 3.5])

# 创建多项式曲线拟合实例
degree = 3
regularization_param = 0.01
curve_fitting = PolynomialCurveFitting(degree, regularization_param)

# 拟合模型
curve_fitting.fit(x, t)

# 进行预测
predictions = curve_fitting.predict(x)

# 计算并打印均方根误差
rmse = curve_fitting.calculate_rmse(x, t)
print(f"均方根误差: {rmse:.4f}")

# 打印拟合的多项式系数
print("拟合的多项式系数:", curve_fitting.coefficients)
