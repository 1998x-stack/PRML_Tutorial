# 00_2.2.1_The_Dirichlet_distribution

"""
Lecture: 2_Probability_Distributions/2.2_Multinomial_Variables
Content: 00_2.2.1_The_Dirichlet_distribution
"""

import numpy as np
import scipy.special as sp

class DirichletDistribution:
    """狄利克雷分布类，用于表示和计算狄利克雷分布的相关属性。

    参数:
    alpha (List[float]): 狄利克雷分布的形状参数向量alpha，每个元素必须大于0。

    示例:
    >>> dirichlet_dist = DirichletDistribution([2, 3, 5])
    >>> pdf_value = dirichlet_dist.pdf([0.2, 0.3, 0.5])
    >>> mean_value = dirichlet_dist.mean()
    >>> var_value = dirichlet_dist.variance()
    >>> cov_value = dirichlet_dist.covariance()
    """
    def __init__(self, alpha: list):
        if any(a <= 0 for a in alpha):
            raise ValueError("参数alpha中的每个元素必须大于0")
        self.alpha = np.array(alpha)
        self.alpha_0 = np.sum(self.alpha)

    def pdf(self, mu: list) -> float:
        """计算给定mu下的狄利克雷分布概率密度函数值。

        参数:
        mu (List[float]): 自变量mu，每个元素在[0, 1]范围内，且元素之和为1。

        返回:
        float: 狄利克雷分布在mu处的概率密度函数值。
        """
        if any(m < 0 or m > 1 for m in mu) or not np.isclose(np.sum(mu), 1):
            raise ValueError("自变量mu的每个元素必须在[0, 1]范围内，且元素之和为1")
        coef = sp.gamma(self.alpha_0) / np.prod(sp.gamma(self.alpha))
        return coef * np.prod([m ** (a - 1) for m, a in zip(mu, self.alpha)])

    def mean(self) -> np.ndarray:
        """计算狄利克雷分布的均值。

        返回:
        np.ndarray: 狄利克雷分布的均值。
        """
        return self.alpha / self.alpha_0

    def variance(self) -> np.ndarray:
        """计算狄利克雷分布的方差。

        返回:
        np.ndarray: 狄利克雷分布的方差。
        """
        return (self.alpha * (self.alpha_0 - self.alpha)) / (self.alpha_0 ** 2 * (self.alpha_0 + 1))

    def covariance(self) -> np.ndarray:
        """计算狄利克雷分布的协方差矩阵。

        返回:
        np.ndarray: 狄利克雷分布的协方差矩阵。
        """
        cov_matrix = np.zeros((len(self.alpha), len(self.alpha)))
        for i in range(len(self.alpha)):
            for j in range(len(self.alpha)):
                if i == j:
                    cov_matrix[i, j] = self.variance()[i]
                else:
                    cov_matrix[i, j] = -self.alpha[i] * self.alpha[j] / (self.alpha_0 ** 2 * (self.alpha_0 + 1))
        return cov_matrix

# 示例使用
if __name__ == "__main__":
    dirichlet_dist = DirichletDistribution([2, 3, 5])
    mu = [0.2, 0.3, 0.5]
    print(f"PDF at mu={mu}: {dirichlet_dist.pdf(mu)}")
    print(f"Mean: {dirichlet_dist.mean()}")
    print(f"Variance: {dirichlet_dist.variance()}")
    print(f"Covariance: {dirichlet_dist.covariance()}")
