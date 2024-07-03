# 00_2.3.1_Conditional_Gaussian_distributions

"""
Lecture: 2_Probability_Distributions/2.3_The_Gaussian_Distribution
Content: 00_2.3.1_Conditional_Gaussian_distributions
"""

import numpy as np
from scipy.stats import multivariate_normal
from typing import Tuple

class ConditionalGaussian:
    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        """
        初始化条件高斯分布类
        
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
    
    def conditional_distribution(self, indices_a: np.ndarray, indices_b: np.ndarray, x_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算条件高斯分布的均值和协方差矩阵
        
        参数:
        indices_a (np.ndarray): 子集A的索引
        indices_b (np.ndarray): 子集B的索引
        x_b (np.ndarray): 给定的子集B的值
        
        返回:
        Tuple[np.ndarray, np.ndarray]: 条件高斯分布的均值和协方差矩阵
        """
        mu_a = self.mu[indices_a]
        mu_b = self.mu[indices_b]
        sigma_aa = self.sigma[np.ix_(indices_a, indices_a)]
        sigma_ab = self.sigma[np.ix_(indices_a, indices_b)]
        sigma_bb = self.sigma[np.ix_(indices_b, indices_b)]
        sigma_ba = self.sigma[np.ix_(indices_b, indices_a)]

        sigma_bb_inv = np.linalg.inv(sigma_bb)
        mu_cond = mu_a + sigma_ab @ sigma_bb_inv @ (x_b - mu_b)
        sigma_cond = sigma_aa - sigma_ab @ sigma_bb_inv @ sigma_ba

        return mu_cond, sigma_cond
    
    def sample_conditional(self, indices_a: np.ndarray, indices_b: np.ndarray, x_b: np.ndarray, size: int = 1) -> np.ndarray:
        """
        从条件高斯分布中采样
        
        参数:
        indices_a (np.ndarray): 子集A的索引
        indices_b (np.ndarray): 子集B的索引
        x_b (np.ndarray): 给定的子集B的值
        size (int): 采样数量
        
        返回:
        np.ndarray: 采样结果
        """
        mu_cond, sigma_cond = self.conditional_distribution(indices_a, indices_b, x_b)
        samples = multivariate_normal.rvs(mean=mu_cond, cov=sigma_cond, size=size)
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
    
    cg = ConditionalGaussian(mu, sigma)
    indices_a = np.array([0, 1])
    indices_b = np.array([2, 3])
    x_b = np.array([2.5, 3.5])
    
    mu_cond, sigma_cond = cg.conditional_distribution(indices_a, indices_b, x_b)
    print("条件均值:", mu_cond)
    print("条件协方差矩阵:", sigma_cond)
    
    samples = cg.sample_conditional(indices_a, indices_b, x_b, size=5)
    print("条件采样结果:", samples)