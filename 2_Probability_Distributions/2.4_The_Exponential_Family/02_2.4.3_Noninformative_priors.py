# 02_2.4.3_Noninformative_priors

"""
Lecture: 2_Probability_Distributions/2.4_The_Exponential_Family
Content: 02_2.4.3_Noninformative_priors
"""

import numpy as np
from scipy.stats import norm, invgamma
from typing import Tuple

class NoninformativePriors:
    def __init__(self, data: np.ndarray):
        """
        初始化非信息先验类
        
        参数:
        data (np.ndarray): 数据集，每行为一个样本点
        """
        self.data = data
        self.n_samples = data.shape[0]
        self.mean = np.mean(data)
        self.var = np.var(data)

    def jeffreys_prior(self) -> Tuple[float, float]:
        """
        杰弗里斯先验
        
        返回:
        Tuple[float, float]: 后验均值和方差
        """
        posterior_mean = self.mean
        posterior_var = self.var / self.n_samples
        return posterior_mean, posterior_var

    def noninformative_prior_mean(self) -> Tuple[float, float]:
        """
        均值参数的非信息先验
        
        返回:
        Tuple[float, float]: 后验均值和方差
        """
        posterior_mean = self.mean
        posterior_var = self.var / self.n_samples
        return posterior_mean, posterior_var

    def noninformative_prior_scale(self) -> Tuple[float, float]:
        """
        尺度参数的非信息先验
        
        返回:
        Tuple[float, float]: 后验尺度参数的alpha和beta
        """
        alpha_post = (self.n_samples - 1) / 2
        beta_post = np.sum((self.data - self.mean)**2) / 2
        return alpha_post, beta_post

# 示例用法
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(0)
    data = np.random.normal(0, 1, 100)
    
    # 创建非信息先验类
    noninformative_priors = NoninformativePriors(data)
    
    # 杰弗里斯先验
    jeffreys_mean, jeffreys_var = noninformative_priors.jeffreys_prior()
    print("杰弗里斯先验后验均值:", jeffreys_mean, "后验方差:", jeffreys_var)
    
    # 均值参数的非信息先验
    noninform_mean, noninform_var = noninformative_priors.noninformative_prior_mean()
    print("非信息先验后验均值:", noninform_mean, "后验方差:", noninform_var)
    
    # 尺度参数的非信息先验
    alpha_post, beta_post = noninformative_priors.noninformative_prior_scale()
    print("非信息先验后验alpha:", alpha_post, "后验beta:", beta_post)