# 05_4.3.6_Canonical_link_functions

"""
Lecture: 4_Linear_Models_for_Classification/4.3_Probabilistic_Discriminative_Models
Content: 05_4.3.6_Canonical_link_functions
"""

import numpy as np
from scipy.special import expit, logit, softmax
from scipy.optimize import minimize
from typing import Tuple

class GeneralizedLinearModel:
    """
    广义线性模型，支持线性回归、逻辑回归和多类别逻辑回归

    Parameters:
    -----------
    link_function : str
        链接函数类型 ('identity', 'logit', 'log_ratio')
    max_iter : int
        训练数据迭代次数 (默认值为 100)
    tol : float
        收敛阈值 (默认值为 1e-6)
    
    Attributes:
    -----------
    w_ : np.ndarray
        权重向量
    """
    def __init__(self, link_function: str, max_iter: int = 100, tol: float = 1e-6) -> None:
        self.link_function = link_function
        self.max_iter = max_iter
        self.tol = tol
        self.w_ = None

    def _identity(self, X: np.ndarray) -> np.ndarray:
        """ 恒等函数 """
        return X

    def _inverse_identity(self, X: np.ndarray) -> np.ndarray:
        """ 恒等函数的逆函数 """
        return X

    def _logit(self, X: np.ndarray) -> np.ndarray:
        """ 对数几率函数 """
        return logit(X)

    def _inverse_logit(self, X: np.ndarray) -> np.ndarray:
        """ 对数几率函数的逆函数 """
        return expit(X)

    def _log_ratio(self, X: np.ndarray) -> np.ndarray:
        """ 对数比率函数 """
        return np.log(X / (1 - X))

    def _inverse_log_ratio(self, X: np.ndarray) -> np.ndarray:
        """ 对数比率函数的逆函数 """
        return np.exp(X) / (1 + np.exp(X))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练广义线性模型
        
        Parameters:
        -----------
        X : np.ndarray, shape = [n_samples, n_features]
            训练向量
        y : np.ndarray, shape = [n_samples]
            目标值
        """
        n_samples, n_features = X.shape
        X_bias = np.insert(X, 0, 1, axis=1)  # 添加偏置项

        if self.link_function == 'identity':
            link, inverse_link = self._identity, self._inverse_identity
        elif self.link_function == 'logit':
            link, inverse_link = self._logit, self._inverse_logit
        elif self.link_function == 'log_ratio':
            link, inverse_link = self._log_ratio, self._inverse_log_ratio
        else:
            raise ValueError("Unsupported link function")

        def neg_log_likelihood(w):
            z = np.dot(X_bias, w)
            mu = inverse_link(z)
            if self.link_function == 'identity':
                likelihood = -0.5 * np.sum((y - mu) ** 2)
            elif self.link_function in ['logit', 'log_ratio']:
                likelihood = y * np.log(mu) + (1 - y) * np.log(1 - mu)
            return -np.sum(likelihood)

        def grad_neg_log_likelihood(w):
            z = np.dot(X_bias, w)
            mu = inverse_link(z)
            if self.link_function == 'identity':
                gradient = np.dot(X_bias.T, y - mu)
            elif self.link_function in ['logit', 'log_ratio']:
                gradient = np.dot(X_bias.T, y - mu)
            return -gradient

        self.w_ = minimize(neg_log_likelihood, np.zeros(n_features + 1), jac=grad_neg_log_likelihood,
                           options={'maxiter': self.max_iter, 'disp': True, 'gtol': self.tol}).x

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        返回预测值
        
        Parameters:
        -----------
        X : np.ndarray
            输入向量
        
        Returns:
        --------
        np.ndarray
            预测值
        """
        X_bias = np.insert(X, 0, 1, axis=1)  # 添加偏置项
        z = np.dot(X_bias, self.w_)
        if self.link_function == 'identity':
            return self._inverse_identity(z)
        elif self.link_function == 'logit':
            return self._inverse_logit(z)
        elif self.link_function == 'log_ratio':
            return self._inverse_log_ratio(z)

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
    主函数，运行广义线性模型并打印结果
    """
    X, y = generate_data()
    glm = GeneralizedLinearModel(link_function='logit', max_iter=100, tol=1e-6)
    glm.fit(X, y)
    predictions = glm.predict(X)
    
    print("权重向量 w:")
    print(glm.w_)
    print("预测结果:")
    print(predictions)

if __name__ == "__main__":
    main()
