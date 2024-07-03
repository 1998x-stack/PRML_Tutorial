# 05_2.3.6_Bayesian_inference_for_the_Gaussian

"""
Lecture: 2_Probability_Distributions/2.3_The_Gaussian_Distribution
Content: 05_2.3.6_Bayesian_inference_for_the_Gaussian
"""

import numpy as np
from typing import Tuple

class BayesianGaussianInference:
    def __init__(self, sigma: float, mu_0: float, sigma_0: float):
        """
        初始化高斯分布贝叶斯推断类
        
        参数:
        sigma (float): 数据分布的已知标准差
        mu_0 (float): 均值的先验分布的均值
        sigma_0 (float): 均值的先验分布的标准差
        """
        self.sigma = sigma
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0
        self.mu_N = mu_0
        self.sigma_N2 = sigma_0 ** 2
    
    def update(self, data: np.ndarray) -> Tuple[float, float]:
        """
        使用新的数据更新后验分布的均值和方差
        
        参数:
        data (np.ndarray): 新的数据点集合
        
        返回:
        Tuple[float, float]: 更新后的后验分布的均值和方差
        """
        n = len(data)
        sum_data = np.sum(data)
        
        # 更新后验分布的均值
        self.mu_N = (self.sigma_0 ** 2 * sum_data + self.sigma ** 2 * self.mu_0) / (n * self.sigma_0 ** 2 + self.sigma ** 2)
        
        # 更新后验分布的方差
        self.sigma_N2 = (1 / self.sigma_0 ** 2 + n / self.sigma ** 2) ** -1
        
        return self.mu_N, np.sqrt(self.sigma_N2)
    
    def get_posterior(self) -> Tuple[float, float]:
        """
        获取当前的后验分布的均值和方差
        
        返回:
        Tuple[float, float]: 当前的后验分布的均值和方差
        """
        return self.mu_N, np.sqrt(self.sigma_N2)

# 示例用法
if __name__ == "__main__":
    # 已知参数
    sigma = 1.0
    mu_0 = 0.0
    sigma_0 = 1.0
    
    # 生成示例数据
    np.random.seed(0)
    data = np.random.normal(loc=0.5, scale=sigma, size=10)
    
    # 创建贝叶斯推断类
    bayesian_inference = BayesianGaussianInference(sigma, mu_0, sigma_0)
    
    # 更新后验分布
    mu_N, sigma_N = bayesian_inference.update(data)
    
    print("更新后的后验均值:", mu_N)
    print("更新后的后验标准差:", sigma_N)
    
    # 获取当前后验分布
    current_mu_N, current_sigma_N = bayesian_inference.get_posterior()
    print("当前的后验均值:", current_mu_N)
    print("当前的后验标准差:", current_sigma_N)