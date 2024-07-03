# 04_2.3.5_Sequential_estimation

"""
Lecture: 2_Probability_Distributions/2.3_The_Gaussian_Distribution
Content: 04_2.3.5_Sequential_estimation
"""

import numpy as np
from typing import Tuple

class GaussianSequentialEstimator:
    def __init__(self, n_features: int):
        """
        初始化高斯分布的序列估计类
        
        参数:
        n_features (int): 数据的特征数量
        """
        self.n_features = n_features
        self.n_samples = 0
        self.mu_ml = np.zeros(n_features)
        self.sigma_ml = np.zeros((n_features, n_features))
    
    def update(self, x: np.ndarray) -> None:
        """
        使用新的数据点更新均值向量和协方差矩阵
        
        参数:
        x (np.ndarray): 新的数据点
        """
        assert x.shape[0] == self.n_features, "数据点的特征数量应与初始化时指定的一致"
        
        self.n_samples += 1
        if self.n_samples == 1:
            self.mu_ml = x
            self.sigma_ml = np.zeros((self.n_features, self.n_features))
        else:
            prev_mu_ml = self.mu_ml.copy()
            self.mu_ml += (x - self.mu_ml) / self.n_samples
            self.sigma_ml = ((self.n_samples - 1) * self.sigma_ml + np.outer(x - prev_mu_ml, x - self.mu_ml)) / self.n_samples
    
    def get_estimates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取当前的均值向量和协方差矩阵估计值
        
        返回:
        Tuple[np.ndarray, np.ndarray]: 均值向量和协方差矩阵
        """
        return self.mu_ml, self.sigma_ml

# 示例用法
if __name__ == "__main__":
    np.random.seed(0)
    n_features = 2
    data = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=100)
    
    estimator = GaussianSequentialEstimator(n_features)
    
    for x in data:
        estimator.update(x)
    
    mu_ml, sigma_ml = estimator.get_estimates()
    
    print("均值的最大似然估计:", mu_ml)
    print("协方差矩阵的最大似然估计:\n", sigma_ml)