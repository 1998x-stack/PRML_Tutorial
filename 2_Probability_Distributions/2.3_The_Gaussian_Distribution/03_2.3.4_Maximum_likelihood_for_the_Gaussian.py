# 03_2.3.4_Maximum_likelihood_for_the_Gaussian

"""
Lecture: 2_Probability_Distributions/2.3_The_Gaussian_Distribution
Content: 03_2.3.4_Maximum_likelihood_for_the_Gaussian
"""

import numpy as np
from typing import Tuple

class GaussianMLE:
    def __init__(self, data: np.ndarray):
        """
        初始化高斯分布的最大似然估计类
        
        参数:
        data (np.ndarray): 数据集，每行为一个样本点
        """
        self.data = data
        self.n_samples, self.n_features = data.shape
    
    def estimate_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        估计高斯分布的均值向量和协方差矩阵
        
        返回:
        Tuple[np.ndarray, np.ndarray]: 均值向量和协方差矩阵
        """
        # 计算均值向量
        mu_ml = np.mean(self.data, axis=0)
        
        # 计算协方差矩阵
        centered_data = self.data - mu_ml
        sigma_ml = np.dot(centered_data.T, centered_data) / self.n_samples
        
        return mu_ml, sigma_ml

# 示例用法
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(0)
    data = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=100)
    
    # 创建MLE估计类
    mle = GaussianMLE(data)
    
    # 估计参数
    mu_ml, sigma_ml = mle.estimate_parameters()
    
    print("均值的最大似然估计:", mu_ml)
    print("协方差矩阵的最大似然估计:\n", sigma_ml)