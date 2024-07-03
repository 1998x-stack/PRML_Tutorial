# 01_3.1.2_Geometry_of_least_squares

"""
Lecture: 3_Linear_Models_for_Regression/3.1_Linear_Basis_Function_Models
Content: 01_3.1.2_Geometry_of_least_squares
"""

import numpy as np
from scipy.linalg import svd

class LinearRegression:
    """
    使用最小二乘法进行线性回归的实现
    
    Attributes:
        weights (np.ndarray): 线性回归模型的权重
    """
    
    def __init__(self):
        self.weights = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        拟合线性回归模型
        
        Args:
            X (np.ndarray): 训练数据集的特征矩阵
            y (np.ndarray): 训练数据集的目标变量向量
        """
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])  # 添加偏置项
        self.weights = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y  # 使用伪逆计算权重
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用拟合好的模型进行预测
        
        Args:
            X (np.ndarray): 测试数据集的特征矩阵
            
        Returns:
            np.ndarray: 预测的结果
        """
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])  # 添加偏置项
        return X_bias @ self.weights

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        计算模型的决定系数 R^2
        
        Args:
            X (np.ndarray): 测试数据集的特征矩阵
            y (np.ndarray): 测试数据集的目标变量向量
            
        Returns:
            float: 模型的 R^2 值
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(0)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    
    # 拟合模型
    model = LinearRegression()
    model.fit(X, y)
    
    # 打印权重
    print("模型权重:", model.weights)
    
    # 预测
    X_new = np.array([[0], [2]])
    y_pred = model.predict(X_new)
    print("预测结果:", y_pred)
    
    # 计算 R^2 值
    r2_score = model.score(X, y)
    print("模型的 R^2 值:", r2_score)
