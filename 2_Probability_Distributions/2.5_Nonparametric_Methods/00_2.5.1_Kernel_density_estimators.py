# 00_2.5.1_Kernel_density_estimators

"""
Lecture: 2_Probability_Distributions/2.5_Nonparametric_Methods
Content: 00_2.5.1_Kernel_density_estimators
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, List

class KernelDensityEstimator:
    def __init__(self, bandwidth: float, kernel: str = 'gaussian'):
        """
        初始化核密度估计类
        
        参数:
        bandwidth (float): 带宽参数
        kernel (str): 核函数类型，默认为'gaussian'
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
    
    def _kernel_function(self, u: np.ndarray) -> np.ndarray:
        """
        核函数
        
        参数:
        u (np.ndarray): 标准化数据
        
        返回:
        np.ndarray: 核函数值
        """
        if self.kernel == 'gaussian':
            return norm.pdf(u)
        else:
            raise ValueError(f"不支持的核函数类型: {self.kernel}")
    
    def fit(self, data: np.ndarray) -> None:
        """
        拟合核密度估计器
        
        参数:
        data (np.ndarray): 输入数据
        """
        self.data = data
        self.n_samples, self.n_features = data.shape
    
    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """
        评估核密度估计
        
        参数:
        points (np.ndarray): 评估点
        
        返回:
        np.ndarray: 评估点的密度值
        """
        n_points = points.shape[0]
        densities = np.zeros(n_points)
        for i, point in enumerate(points):
            diff = self.data - point
            u = diff / self.bandwidth
            kernels = self._kernel_function(u)
            densities[i] = np.sum(kernels) / (self.n_samples * (self.bandwidth ** self.n_features))
        return densities

# 示例用法
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(0)
    data = np.random.normal(0, 1, (100, 1))
    
    # 创建核密度估计器
    kde = KernelDensityEstimator(bandwidth=0.5)
    kde.fit(data)
    
    # 评估点
    points = np.linspace(-3, 3, 100).reshape(-1, 1)
    densities = kde.evaluate(points)
    
    print("评估点的密度值:", densities)