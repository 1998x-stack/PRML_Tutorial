# 01_2.3.2_Marginal_Gaussian_distributions

"""
Lecture: 2_Probability_Distributions/2.3_The_Gaussian_Distribution
Content: 01_2.3.2_Marginal_Gaussian_distributions
"""

import numpy as np
from scipy.stats import multivariate_normal
from typing import Tuple

class MarginalGaussian:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        """
        初始化边缘高斯分布类
        
        参数:
        mu (np.ndarray): 均值向量
        sigma (np.ndarray): 协方差矩阵
        """
        self.mu = mu
        self.sigma = sigma
        self._check_validity()
    
    def _check_validity(self):
        """检查均值向量和协方差矩阵的有效性"""
        assert self.mu.ndim == 1, "均值向量应为一维"
        assert self.sigma.ndim == 2, "协方差矩阵应为二维"
        assert self.sigma.shape[0] == self.sigma.shape[1], "协方差矩阵应为方阵"
        assert self.mu.shape[0] == self.sigma.shape[0], "均值向量和协方差矩阵的维度应匹配"
    
    def marginal_distribution(self, indices_a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算边缘高斯分布的均值和协方差矩阵
        
        参数:
        indices_a (np.ndarray): 子集A的索引
        
        返回:
        Tuple[np.ndarray, np.ndarray]: 边缘高斯分布的均值和协方差矩阵
        """
        mu_a = self.mu[indices_a]
        sigma_aa = self.sigma[np.ix_(indices_a, indices_a)]

        return mu_a, sigma_aa
    
    def sample_marginal(self, indices_a: np.ndarray, size: int = 1) -> np.ndarray:
        """
        从边缘高斯分布中采样
        
        参数:
        indices_a (np.ndarray): 子集A的索引
        size (int): 采样数量
        
        返回:
        np.ndarray: 采样结果
        """
        mu_a, sigma_aa = self.marginal_distribution(indices_a)
        samples = multivariate_normal.rvs(mean=mu_a, cov=sigma_aa, size=size)
        return samples

# 示例用法
if __name__ == "__main__":
    mu = np.array([1.0, 2.0, 3.0, 4.0])
    sigma = np.array([
        [1.0, 0.5, 0.3, 0.2],
        [0.5, 1.0, 0.4, 0.3],
        [0.3, 0.4, 1.0, 0.6],
        [0.2, 0.3, 0.6, 1.0]
    ])
    
    mg = MarginalGaussian(mu, sigma)
    indices_a = np.array([0, 1])
    
    mu_a, sigma_aa = mg.marginal_distribution(indices_a)
    print("边缘均值:", mu_a)
    print("边缘协方差矩阵:", sigma_aa)
    
    samples = mg.sample_marginal(indices_a, size=5)
    print("边缘采样结果:", samples)
