# 01_2.4.2_Conjugate_priors

"""
Lecture: 2_Probability_Distributions/2.4_The_Exponential_Family
Content: 01_2.4.2_Conjugate_priors
"""

import numpy as np
from scipy.stats import beta, norm, invgamma
from typing import Tuple, List

class ConjugatePriors:
    def __init__(self, data: np.ndarray):
        """
        初始化共轭先验类
        
        参数:
        data (np.ndarray): 数据集，每行为一个样本点
        """
        self.data = data
        self.n_samples, self.n_features = data.shape
    
    def beta_prior(self, alpha: float, beta: float) -> Tuple[float, float]:
        """
        贝塔分布作为共轭先验
        
        参数:
        alpha (float): 贝塔分布的α参数
        beta (float): 贝塔分布的β参数
        
        返回:
        Tuple[float, float]: 更新后的α和β参数
        """
        success = np.sum(self.data)
        failure = self.n_samples - success
        alpha_post = alpha + success
        beta_post = beta + failure
        return alpha_post, beta_post
    
    def gaussian_prior(self, mu_0: float, lambda_0: float, sigma_0: float) -> Tuple[float, float, float]:
        """
        高斯分布作为共轭先验
        
        参数:
        mu_0 (float): 先验均值
        lambda_0 (float): 先验精度
        sigma_0 (float): 先验标准差
        
        返回:
        Tuple[float, float, float]: 更新后的均值、精度和标准差
        """
        n = self.n_samples
        sample_mean = np.mean(self.data)
        sample_var = np.var(self.data)
        
        mu_post = (lambda_0 * mu_0 + n * sample_mean) / (lambda_0 + n)
        lambda_post = lambda_0 + n
        sigma_post = np.sqrt((lambda_0 * sigma_0**2 + (n - 1) * sample_var + n * lambda_0 * (sample_mean - mu_0)**2 / (lambda_0 + n)) / (lambda_0 + n))
        
        return mu_post, lambda_post, sigma_post
    
    def wishart_prior(self, nu_0: float, W_0: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Wishart分布作为共轭先验
        
        参数:
        nu_0 (float): 自由度参数
        W_0 (np.ndarray): 逆规模矩阵
        
        返回:
        Tuple[float, np.ndarray]: 更新后的自由度参数和逆规模矩阵
        """
        n = self.n_samples
        nu_post = nu_0 + n
        S = np.dot(self.data.T, self.data)
        W_post = np.linalg.inv(np.linalg.inv(W_0) + S)
        
        return nu_post, W_post

# 示例用法
if __name__ == "__main__":
    # 生成伯努利分布示例数据
    np.random.seed(0)
    data_bernoulli = np.random.binomial(1, 0.6, 100)
    
    # 贝塔分布先验
    conjugate_priors = ConjugatePriors(data_bernoulli)
    alpha_post, beta_post = conjugate_priors.beta_prior(alpha=2, beta=2)
    print("贝塔分布后验参数: alpha =", alpha_post, ", beta =", beta_post)
    
    # 生成高斯分布示例数据
    data_gaussian = np.random.normal(0, 1, 100)
    
    # 高斯分布先验
    conjugate_priors = ConjugatePriors(data_gaussian)
    mu_post, lambda_post, sigma_post = conjugate_priors.gaussian_prior(mu_0=0, lambda_0=1, sigma_0=1)
    print("高斯分布后验参数: mu =", mu_post, ", lambda =", lambda_post, ", sigma =", sigma_post)
    
    # 生成多变量高斯分布示例数据
    data_multivariate = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)
    
    # Wishart分布先验
    conjugate_priors = ConjugatePriors(data_multivariate)
    nu_post, W_post = conjugate_priors.wishart_prior(nu_0=2, W_0=np.eye(2))
    print("Wishart分布后验参数: nu =", nu_post)
    print("W_post =", W_post)