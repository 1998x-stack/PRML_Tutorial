# 03_3.1.4_Regularized_least_squares

"""
Lecture: 3_Linear_Models_for_Regression/3.1_Linear_Basis_Function_Models
Content: 03_3.1.4_Regularized_least_squares
"""

import numpy as np
from typing import Tuple

class RegularizedLeastSquares:
    def __init__(self, lambda_: float):
        """
        初始化正则化最小二乘类

        参数:
        lambda_ (float): 正则化系数
        """
        self.lambda_ = lambda_
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        拟合正则化最小二乘模型

        参数:
        X (np.ndarray): 训练数据集的特征矩阵
        y (np.ndarray): 训练数据集的目标变量向量
        """
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])  # 添加偏置项
        I = np.eye(X_bias.shape[1])
        self.weights = np.linalg.inv(self.lambda_ * I + X_bias.T @ X_bias) @ X_bias.T @ y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用拟合好的模型进行预测

        参数:
        X (np.ndarray): 测试数据集的特征矩阵

        返回:
        np.ndarray: 预测的结果
        """
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])  # 添加偏置项
        return X_bias @ self.weights

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        评估模型性能

        参数:
        X (np.ndarray): 测试数据集的特征矩阵
        y (np.ndarray): 测试数据集的目标变量向量

        返回:
        float: 平均平方误差
        """
        predictions = self.predict(X)
        errors = y - predictions
        return np.mean(errors ** 2)

# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(0)
    X_train = 2 * np.random.rand(100, 1)
    y_train = 4 + 3 * X_train + np.random.randn(100, 1)

    # 初始化和训练模型
    lambda_ = 0.1
    model = RegularizedLeastSquares(lambda_=lambda_)
    model.fit(X_train, y_train)

    # 打印权重
    print("模型权重:", model.weights)

    # 预测
    X_test = np.array([[0], [2]])
    y_pred = model.predict(X_test)
    print("预测结果:", y_pred)

    # 评估模型性能
    mse = model.evaluate(X_train, y_train)
    print("平均平方误差:", mse)