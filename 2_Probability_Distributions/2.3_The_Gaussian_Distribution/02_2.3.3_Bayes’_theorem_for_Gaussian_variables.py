# 02_2.3.3_Bayes’_theorem_for_Gaussian_variables

"""
Lecture: 2_Probability_Distributions/2.3_The_Gaussian_Distribution
Content: 02_2.3.3_Bayes’_theorem_for_Gaussian_variables
"""

import numpy as np
from scipy.stats import multivariate_normal
from typing import Tuple

class BayesianGaussian:
    def __init__(self, mu: np.ndarray, lambda_: np.ndarray, a: np.ndarray, b: np.ndarray, l: np.ndarray):
        """
        初始化贝叶斯高斯分布类
        
        参数:
        mu (np.ndarray): 边缘分布的均值向量
        lambda_ (np.ndarray): 边缘分布的精度矩阵
        a (np.ndarray): 条件分布的均值矩阵系数
        b (np.ndarray): 条件分布的均值偏移向量
        l (np.ndarray): 条件分布的精度矩阵
        """
        self.mu = mu
        self.lambda_ = lambda_
        self.a = a
        self.b = b
        self.l = l
        self._check_validity()
    
    def _check_validity(self):
        """检查输入参数的有效性"""
        assert self.mu.ndim == 1, "均值向量应为一维"
        assert self.lambda_.ndim == 2, "精度矩阵应为二维"
        assert self.lambda_.shape[0] == self.lambda_.shape[1], "精度矩阵应为方阵"
        assert self.mu.shape[0] == self.lambda_.shape[0], "均值向量和精度矩阵的维度应匹配"
        assert self.a.ndim == 2, "A矩阵应为二维"
        assert self.b.ndim == 1, "b向量应为一维"
        assert self.l.ndim == 2, "L矩阵应为二维"
        assert self.l.shape[0] == self.l.shape[1], "L矩阵应为方阵"
    
    def marginal_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算边缘高斯分布的均值和协方差矩阵
        
        返回:
        Tuple[np.ndarray, np.ndarray]: 边缘高斯分布的均值和协方差矩阵
        """
        mu_y = self.a @ self.mu + self.b
        sigma_y = np.linalg.inv(self.l) + self.a @ np.linalg.inv(self.lambda_) @ self.a.T

        return mu_y, sigma_y
    
    def conditional_distribution(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算条件高斯分布的均值和协方差矩阵
        
        参数:
        y (np.ndarray): 条件变量的值
        
        返回:
        Tuple[np.ndarray, np.ndarray]: 条件高斯分布的均值和协方差矩阵
        """
        sigma = np.linalg.inv(self.lambda_ + self.a.T @ self.l @ self.a)
        mu_x_given_y = sigma @ (self.a.T @ self.l @ (y - self.b) + self.lambda_ @ self.mu)
        
        return mu_x_given_y, sigma
    
    def sample_marginal(self, size: int = 1) -> np.ndarray:
        """
        从边缘高斯分布中采样
        
        参数:
        size (int): 采样数量
        
        返回:
        np.ndarray: 采样结果
        """
        mu_y, sigma_y = self.marginal_distribution()
        samples = multivariate_normal.rvs(mean=mu_y, cov=sigma_y, size=size)
        return samples
    
    def sample_conditional(self, y: np.ndarray, size: int = 1) -> np.ndarray:
        """
        从条件高斯分布中采样
        
        参数:
        y (np.ndarray): 条件变量的值
        size (int): 采样数量
        
        返回:
        np.ndarray: 采样结果
        """
        mu_x_given_y, sigma_x_given_y = self.conditional_distribution(y)
        samples = multivariate_normal.rvs(mean=mu_x_given_y, cov=sigma_x_given_y, size=size)
        return samples

# 示例用法
if __name__ == "__main__":
    mu = np.array([1.0, 2.0])
    lambda_ = np.array([
        [1.0, 0.5],
        [0.5, 1.0]
    ])
    a = np.array([
        [0.6, 0.8],
        [0.3, 0.7]
    ])
    b = np.array([0.5, 0.2])
    l = np.array([
        [2.0, 0.4],
        [0.4, 2.0]
    ])
    
    bg = BayesianGaussian(mu, lambda_, a, b, l)
    
    mu_y, sigma_y = bg.marginal_distribution()
    print("边缘均值:", mu_y)
    print("边缘协方差矩阵:", sigma_y)
    
    y = np.array([1.5, 2.5])
    mu_x_given_y, sigma_x_given_y = bg.conditional_distribution(y)
    print("条件均值:", mu_x_given_y)
    print("条件协方差矩阵:", sigma_x_given_y)
    
    samples_marginal = bg.sample_marginal(size=5)
    print("边缘采样结果:", samples_marginal)
    
    samples_conditional = bg.sample_conditional(y, size=5)
    print("条件采样结果:", samples_conditional)
