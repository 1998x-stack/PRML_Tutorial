# 05_3.6_Limitations_of_Fixed_Basis_Functions

"""
Lecture: /3_Linear_Models_for_Regression
Content: 05_3.6_Limitations_of_Fixed_Basis_Functions
"""

import numpy as np
from scipy.linalg import solve

class FixedBasisLinearModel:
    """
    固定基函数的线性模型
    """
    
    def __init__(self, basis_funcs):
        """
        初始化模型
        
        参数:
            basis_funcs (list): 包含基函数的列表，每个基函数都是一个函数
        """
        self.basis_funcs = basis_funcs
        self.weights = None

    def fit(self, X: np.ndarray, t: np.ndarray):
        """
        拟合模型
        
        参数:
            X (np.ndarray): 输入数据
            t (np.ndarray): 目标值
        """
        # 计算设计矩阵
        Phi = self.design_matrix(X)
        # 计算权重
        self.weights = solve(Phi.T @ Phi, Phi.T @ t)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测新数据
        
        参数:
            X (np.ndarray): 输入数据
        
        返回:
            np.ndarray: 预测值
        """
        Phi = self.design_matrix(X)
        return Phi @ self.weights

    def design_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        计算设计矩阵
        
        参数:
            X (np.ndarray): 输入数据
        
        返回:
            np.ndarray: 设计矩阵
        """
        N = X.shape[0]
        M = len(self.basis_funcs)
        Phi = np.zeros((N, M))
        for i, func in enumerate(self.basis_funcs):
            Phi[:, i] = func(X).flatten()
        return Phi

# 示例基函数
def basis_func_1(x):
    return np.exp(-0.5 * (x - 1)**2)

def basis_func_2(x):
    return np.exp(-0.5 * (x + 1)**2)

if __name__ == "__main__":
    # 生成模拟数据
    X_train = np.linspace(-3, 3, 100).reshape(-1, 1)
    t_train = np.sin(X_train) + 0.1 * np.random.randn(100, 1)
    
    # 定义基函数
    basis_funcs = [basis_func_1, basis_func_2]
    
    # 创建并拟合模型
    model = FixedBasisLinearModel(basis_funcs)
    model.fit(X_train, t_train)
    
    # 预测
    X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
    predictions = model.predict(X_test)
    
    print("预测结果: ", predictions.flatten())