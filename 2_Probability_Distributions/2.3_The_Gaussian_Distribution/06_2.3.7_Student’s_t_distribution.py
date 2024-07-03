# 06_2.3.7_Student’s_t-distribution

"""
Lecture: 2_Probability_Distributions/2.3_The_Gaussian_Distribution
Content: 06_2.3.7_Student’s_t-distribution
"""

import numpy as np
from scipy.special import gamma
from typing import Tuple

class StudentsTDistribution:
    def __init__(self, mu: float, lambda_: float, nu: float):
        """
        初始化Student's t分布类
        
        参数:
        mu (float): 均值参数
        lambda_ (float): 精度参数
        nu (float): 自由度参数
        """
        self.mu = mu
        self.lambda_ = lambda_
        self.nu = nu

    def pdf(self, x: float) -> float:
        """
        计算Student's t分布的概率密度函数值
        
        参数:
        x (float): 自变量值
        
        返回:
        float: 概率密度函数值
        """
        coeff = gamma((self.nu + 1) / 2) / (gamma(self.nu / 2) * np.sqrt(self.nu * np.pi) * np.sqrt(1 / self.lambda_))
        exponent = - (self.nu + 1) / 2
        base = 1 + (self.lambda_ * (x - self.mu) ** 2) / self.nu
        return coeff * base ** exponent

class MultivariateStudentsTDistribution:
    def __init__(self, mu: np.ndarray, Lambda: np.ndarray, nu: float):
        """
        初始化多变量Student's t分布类
        
        参数:
        mu (np.ndarray): 均值向量
        Lambda (np.ndarray): 精度矩阵
        nu (float): 自由度参数
        """
        self.mu = mu
        self.Lambda = Lambda
        self.nu = nu
        self.d = mu.shape[0]  # 变量维度
    
    def pdf(self, x: np.ndarray) -> float:
        """
        计算多变量Student's t分布的概率密度函数值
        
        参数:
        x (np.ndarray): 自变量向量
        
        返回:
        float: 概率密度函数值
        """
        delta = x - self.mu
        delta2 = delta.T @ self.Lambda @ delta
        coeff = gamma((self.nu + self.d) / 2) / (gamma(self.nu / 2) * (self.nu * np.pi) ** (self.d / 2) * np.linalg.det(self.Lambda) ** 0.5)
        exponent = - (self.nu + self.d) / 2
        base = 1 + delta2 / self.nu
        return coeff * base ** exponent

# 示例用法
if __name__ == "__main__":
    # 单变量Student's t分布
    t_dist = StudentsTDistribution(mu=0, lambda_=1, nu=5)
    x_values = np.linspace(-5, 5, 100)
    pdf_values = [t_dist.pdf(x) for x in x_values]
    print("单变量Student's t分布的PDF值:", pdf_values)
    
    # 多变量Student's t分布
    mu = np.array([0, 0])
    Lambda = np.array([[1, 0.5], [0.5, 1]])
    nu = 5
    multi_t_dist = MultivariateStudentsTDistribution(mu, Lambda, nu)
    x = np.array([1, 1])
    pdf_value = multi_t_dist.pdf(x)
    print("多变量Student's t分布在点[1, 1]处的PDF值:", pdf_value)