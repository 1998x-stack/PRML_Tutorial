# 08_2.3.9_Mixtures_of_Gaussians

"""
Lecture: 2_Probability_Distributions/2.3_The_Gaussian_Distribution
Content: 08_2.3.9_Mixtures_of_Gaussians
"""

import numpy as np
from scipy.stats import multivariate_normal
from typing import Tuple, List

class GaussianMixtureModel:
    def __init__(self, n_components: int, tol: float = 1e-6, max_iter: int = 100):
        """
        初始化高斯混合模型类
        
        参数:
        n_components (int): 混合分量的数量
        tol (float): 收敛阈值
        max_iter (int): 最大迭代次数
        """
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter

    def _initialize_parameters(self, X: np.ndarray) -> None:
        """
        初始化模型参数
        
        参数:
        X (np.ndarray): 输入数据
        """
        n_samples, n_features = X.shape
        self.weights = np.ones(self.n_components) / self.n_components
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]
        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])
    
    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """
        E步：计算责任值
        
        参数:
        X (np.ndarray): 输入数据
        
        返回:
        np.ndarray: 责任值矩阵
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(X, mean=self.means[k], cov=self.covariances[k])
        sum_responsibilities = responsibilities.sum(axis=1, keepdims=True)
        return responsibilities / sum_responsibilities
    
    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray) -> None:
        """
        M步：更新模型参数
        
        参数:
        X (np.ndarray): 输入数据
        responsibilities (np.ndarray): 责任值矩阵
        """
        n_samples, n_features = X.shape
        for k in range(self.n_components):
            responsibility = responsibilities[:, k]
            total_responsibility = responsibility.sum()
            self.weights[k] = total_responsibility / n_samples
            self.means[k] = (X * responsibility[:, np.newaxis]).sum(axis=0) / total_responsibility
            diff = X - self.means[k]
            self.covariances[k] = np.dot((responsibility[:, np.newaxis] * diff).T, diff) / total_responsibility
    
    def fit(self, X: np.ndarray) -> None:
        """
        训练高斯混合模型
        
        参数:
        X (np.ndarray): 输入数据
        """
        self._initialize_parameters(X)
        log_likelihood = []
        for _ in range(self.max_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)
            log_likelihood.append(np.sum(np.log(np.sum(responsibilities, axis=1))))
            if len(log_likelihood) > 1 and abs(log_likelihood[-1] - log_likelihood[-2]) < self.tol:
                break

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测每个样本点的责任值
        
        参数:
        X (np.ndarray): 输入数据
        
        返回:
        np.ndarray: 每个样本点的责任值矩阵
        """
        return self._e_step(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测每个样本点的簇标签
        
        参数:
        X (np.ndarray): 输入数据
        
        返回:
        np.ndarray: 每个样本点的簇标签
        """
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

# 示例用法
if __name__ == "__main__":
    np.random.seed(0)
    X = np.vstack([
        np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=100),
        np.random.multivariate_normal(mean=[3, 3], cov=[[1, -0.5], [-0.5, 1]], size=100)
    ])
    
    gmm = GaussianMixtureModel(n_components=2)
    gmm.fit(X)
    
    labels = gmm.predict(X)
    print("预测的簇标签:", labels)