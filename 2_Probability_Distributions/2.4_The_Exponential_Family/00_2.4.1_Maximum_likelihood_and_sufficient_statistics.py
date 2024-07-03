# 00_2.4.1_Maximum_likelihood_and_sufficient_statistics

"""
Lecture: 2_Probability_Distributions/2.4_The_Exponential_Family
Content: 00_2.4.1_Maximum_likelihood_and_sufficient_statistics
"""

import numpy as np
from typing import Tuple, List

class ExponentialFamilyMLE:
    def __init__(self, data: np.ndarray):
        """
        初始化指数族分布的最大似然估计类
        
        参数:
        data (np.ndarray): 数据集，每行为一个样本点
        """
        self.data = data
        self.n_samples, self.n_features = data.shape

    def sufficient_statistics(self) -> np.ndarray:
        """
        计算充分统计量
        
        返回:
        np.ndarray: 充分统计量
        """
        return np.sum(self.data, axis=0)

    def log_likelihood(self, eta: np.ndarray) -> float:
        """
        计算对数似然函数
        
        参数:
        eta (np.ndarray): 参数向量
        
        返回:
        float: 对数似然值
        """
        u_x = self.sufficient_statistics()
        log_likelihood = np.dot(eta, u_x) - self.n_samples * self.g_function(eta)
        return log_likelihood

    def g_function(self, eta: np.ndarray) -> float:
        """
        计算g函数
        
        参数:
        eta (np.ndarray): 参数向量
        
        返回:
        float: g函数值
        """
        # 此处假设g函数为简化形式，实际应用中需根据具体分布定义g函数
        return np.sum(np.exp(eta))

    def fit(self, tol: float = 1e-6, max_iter: int = 100) -> np.ndarray:
        """
        使用最大似然估计法拟合模型参数
        
        参数:
        tol (float): 收敛阈值
        max_iter (int): 最大迭代次数
        
        返回:
        np.ndarray: 拟合的参数向量
        """
        eta = np.zeros(self.n_features)
        for _ in range(max_iter):
            eta_new = self.sufficient_statistics() / self.n_samples
            if np.linalg.norm(eta_new - eta) < tol:
                break
            eta = eta_new
        return eta

# 示例用法
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(0)
    data = np.random.rand(100, 2)
    
    # 创建MLE估计类
    mle = ExponentialFamilyMLE(data)
    
    # 拟合模型参数
    eta_mle = mle.fit()
    
    print("最大似然估计的参数:", eta_mle)