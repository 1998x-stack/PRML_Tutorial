# 00_2.1.1_The_beta_distribution

"""
Lecture: 2_Probability_Distributions/2.1_Binary_Variables
Content: 00_2.1.1_The_beta_distribution
"""

import numpy as np
import scipy.special as sp

class BetaDistribution:
    """贝塔分布类，用于表示和计算贝塔分布的相关属性。

    参数:
    a (float): 贝塔分布的形状参数a，a > 0。
    b (float): 贝塔分布的形状参数b，b > 0。

    示例:
    >>> beta_dist = BetaDistribution(2, 3)
    >>> pdf_value = beta_dist.pdf(0.5)
    >>> mean_value = beta_dist.mean()
    >>> var_value = beta_dist.variance()
    """
    def __init__(self, a: float, b: float):
        if a <= 0 or b <= 0:
            raise ValueError("参数a和b必须大于0")
        self.a = a
        self.b = b

    def pdf(self, x: float) -> float:
        """计算给定x下的贝塔分布概率密度函数值。

        参数:
        x (float): 自变量x，0 <= x <= 1。

        返回:
        float: 贝塔分布在x处的概率密度函数值。
        """
        if x < 0 or x > 1:
            raise ValueError("自变量x必须在[0, 1]范围内")
        coef = sp.gamma(self.a + self.b) / (sp.gamma(self.a) * sp.gamma(self.b))
        return coef * (x ** (self.a - 1)) * ((1 - x) ** (self.b - 1))

    def mean(self) -> float:
        """计算贝塔分布的均值。

        返回:
        float: 贝塔分布的均值。
        """
        return self.a / (self.a + self.b)

    def variance(self) -> float:
        """计算贝塔分布的方差。

        返回:
        float: 贝塔分布的方差。
        """
        return (self.a * self.b) / ((self.a + self.b) ** 2 * (self.a + self.b + 1))

    def mode(self) -> float:
        """计算贝塔分布的众数。

        返回:
        float: 贝塔分布的众数。
        """
        if self.a > 1 and self.b > 1:
            return (self.a - 1) / (self.a + self.b - 2)
        else:
            raise ValueError("当a和b都大于1时，众数才定义")

# 示例使用
if __name__ == "__main__":
    beta_dist = BetaDistribution(2, 5)
    x = 0.5
    print(f"PDF at x={x}: {beta_dist.pdf(x)}")
    print(f"Mean: {beta_dist.mean()}")
    print(f"Variance: {beta_dist.variance()}")
    try:
        print(f"Mode: {beta_dist.mode()}")
    except ValueError as e:
        print(e)