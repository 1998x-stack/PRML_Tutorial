# 04_4.3.5_Probit_regression

"""
Lecture: 4_Linear_Models_for_Classification/4.3_Probabilistic_Discriminative_Models
Content: 04_4.3.5_Probit_regression
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Tuple

class ProbitRegression:
    """
    Probit 回归分类器

    Parameters:
    -----------
    max_iter : int
        训练数据迭代次数 (默认值为 100)
    tol : float
        收敛阈值 (默认值为 1e-6)

    Attributes:
    -----------
    w_ : np.ndarray
        权重向量
    """
    def __init__(self, max_iter: int = 100, tol: float = 1e-6) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.w_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练 Probit 回归分类器
        
        Parameters:
        -----------
        X : np.ndarray, shape = [n_samples, n_features]
            训练向量
        y : np.ndarray, shape = [n_samples]
            目标值
        """
        # 初始化权重向量
        self.w_ = np.zeros(X.shape[1] + 1)

        X_bias = np.insert(X, 0, 1, axis=1)  # 添加偏置项

        def neg_log_likelihood(w):
            z = np.dot(X_bias, w)
            likelihood = y * np.log(norm.cdf(z)) + (1 - y) * np.log(1 - norm.cdf(z))
            return -np.sum(likelihood)

        def grad_neg_log_likelihood(w):
            z = np.dot(X_bias, w)
            pdf = norm.pdf(z)
            cdf = norm.cdf(z)
            gradient = np.dot(X_bias.T, (y - cdf) * pdf / (cdf * (1 - cdf)))
            return -np.sum(gradient, axis=1)

        result = minimize(neg_log_likelihood, self.w_, jac=grad_neg_log_likelihood, 
                          options={'maxiter': self.max_iter, 'disp': True, 'gtol': self.tol})
        self.w_ = result.x

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        返回类标预测值
        
        Parameters:
        -----------
        X : np.ndarray
            输入向量
        
        Returns:
        --------
        np.ndarray
            类标预测值
        """
        X_bias = np.insert(X, 0, 1, axis=1)  # 添加偏置项
        z = np.dot(X_bias, self.w_)
        return np.where(norm.cdf(z) >= 0.5, 1, 0)

def generate_data(n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成二分类数据集
    
    Parameters:
    -----------
    n_samples : int
        样本数量 (默认值为 100)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        特征矩阵和目标值向量
    """
    np.random.seed(0)
    X1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    X2 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    X = np.vstack((X1, X2))
    y = np.hstack((np.ones(n_samples // 2), np.zeros(n_samples // 2)))
    return X, y

def main() -> None:
    """
    主函数，运行 Probit 回归并打印结果
    """
    X, y = generate_data()
    probit = ProbitRegression(max_iter=100, tol=1e-6)
    probit.fit(X, y)
    predictions = probit.predict(X)
    
    print("权重向量 w:")
    print(probit.w_)

if __name__ == "__main__":
    main()
