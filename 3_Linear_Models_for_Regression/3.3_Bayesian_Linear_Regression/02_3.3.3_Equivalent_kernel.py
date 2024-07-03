# 02_3.3.3_Equivalent_kernel

"""
Lecture: 3_Linear_Models_for_Regression/3.3_Bayesian_Linear_Regression
Content: 02_3.3.3_Equivalent_kernel
"""

import numpy as np
from scipy.linalg import inv

class BayesianLinearRegression:
    """
    贝叶斯线性回归模型类

    参数:
        alpha (float): 先验分布的方差参数
        beta (float): 噪声精度参数
    """
    
    def __init__(self, alpha: float, beta: float):
        """
        初始化贝叶斯线性回归模型

        参数:
            alpha (float): 先验分布的方差参数
            beta (float): 噪声精度参数
        """
        self.alpha = alpha
        self.beta = beta
        self.m_N = None
        self.S_N = None

    def fit(self, X: np.ndarray, t: np.ndarray):
        """
        拟合贝叶斯线性回归模型

        参数:
            X (np.ndarray): 输入数据矩阵
            t (np.ndarray): 目标值向量
        """
        # 添加偏置项
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        
        # 计算先验分布的协方差矩阵
        S_0_inv = self.alpha * np.eye(X.shape[1])
        
        # 计算后验分布的协方差矩阵
        self.S_N = inv(S_0_inv + self.beta * X.T @ X)
        
        # 计算后验分布的均值向量
        self.m_N = self.beta * self.S_N @ X.T @ t
        
        print(f"后验均值向量: {self.m_N}")
        print(f"后验协方差矩阵: {self.S_N}")

    def predict(self, X_new: np.ndarray):
        """
        使用贝叶斯线性回归模型进行预测

        参数:
            X_new (np.ndarray): 新的输入数据矩阵

        返回:
            均值预测值和预测方差
        """
        # 添加偏置项
        X_new = np.hstack([np.ones((X_new.shape[0], 1)), X_new])
        
        # 预测均值
        y_mean = X_new @ self.m_N
        
        # 预测方差
        y_var = 1 / self.beta + np.sum(X_new @ self.S_N * X_new, axis=1)
        
        return y_mean, y_var

    def equivalent_kernel(self, X_new: np.ndarray):
        """
        计算等效核

        参数:
            X_new (np.ndarray): 新的输入数据矩阵

        返回:
            等效核矩阵
        """
        X_new = np.hstack([np.ones((X_new.shape[0], 1)), X_new])
        kernel = self.beta * X_new @ self.S_N @ X_new.T
        return kernel

if __name__ == "__main__":
    # 示例数据
    X_train = np.array([[0.1], [0.4], [0.7], [1.0]])
    t_train = np.array([1.1, 1.9, 3.0, 4.2])
    
    model = BayesianLinearRegression(alpha=1.0, beta=25.0)
    model.fit(X_train, t_train)
    
    # 新数据进行预测
    X_new = np.array([[0.2], [0.5], [0.8]])
    y_mean, y_var = model.predict(X_new)
    kernel = model.equivalent_kernel(X_new)
    
    print("预测均值: ", y_mean)
    print("预测方差: ", y_var)
    print("等效核矩阵: ", kernel)
