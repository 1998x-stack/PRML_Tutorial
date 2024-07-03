# 07_2.3.8_Periodic_variables

"""
Lecture: 2_Probability_Distributions/2.3_The_Gaussian_Distribution
Content: 07_2.3.8_Periodic_variables
"""

import numpy as np
from scipy.special import i0
from typing import List

class PeriodicVariable:
    def __init__(self, theta: List[float]):
        """
        初始化周期变量类
        
        参数:
        theta (List[float]): 周期变量的观测值列表（以弧度表示）
        """
        self.theta = np.array(theta)
        self.n = len(theta)
    
    def mean_direction(self) -> float:
        """
        计算周期变量的平均方向
        
        返回:
        float: 平均方向（以弧度表示）
        """
        sin_sum = np.sum(np.sin(self.theta))
        cos_sum = np.sum(np.cos(self.theta))
        mean_theta = np.arctan2(sin_sum, cos_sum)
        return mean_theta
    
    def resultant_length(self) -> float:
        """
        计算周期变量的合向量长度
        
        返回:
        float: 合向量长度
        """
        sin_sum = np.sum(np.sin(self.theta))
        cos_sum = np.sum(np.cos(self.theta))
        R = np.sqrt(sin_sum**2 + cos_sum**2) / self.n
        return R

class VonMisesDistribution:
    def __init__(self, mu: float, kappa: float):
        """
        初始化von Mises分布类
        
        参数:
        mu (float): 分布的均值方向（以弧度表示）
        kappa (float): 分布的浓度参数
        """
        self.mu = mu
        self.kappa = kappa
    
    def pdf(self, theta: float) -> float:
        """
        计算von Mises分布的概率密度函数值
        
        参数:
        theta (float): 自变量值（以弧度表示）
        
        返回:
        float: 概率密度函数值
        """
        return np.exp(self.kappa * np.cos(theta - self.mu)) / (2 * np.pi * i0(self.kappa))

# 示例用法
if __name__ == "__main__":
    # 示例数据：风向观测值（以弧度表示）
    theta_samples = [0.1, 0.2, 0.3, 6.1, 6.2, 6.3]  # 注意这里以弧度表示
    periodic_var = PeriodicVariable(theta_samples)
    
    mean_dir = periodic_var.mean_direction()
    resultant_len = periodic_var.resultant_length()
    
    print(f"平均方向: {mean_dir} 弧度")
    print(f"合向量长度: {resultant_len}")
    
    # 示例von Mises分布
    vm_dist = VonMisesDistribution(mu=np.pi, kappa=2)
    theta_test = np.linspace(-np.pi, np.pi, 100)
    pdf_values = [vm_dist.pdf(theta) for theta in theta_test]
    
    print(f"von Mises分布的PDF值: {pdf_values[:10]}...")  # 仅打印前10个值以简化输出