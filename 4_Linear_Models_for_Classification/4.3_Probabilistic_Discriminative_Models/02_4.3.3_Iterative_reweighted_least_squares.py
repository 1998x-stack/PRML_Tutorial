# 02_4.3.3_Iterative_reweighted_least_squares

"""
Lecture: 4_Linear_Models_for_Classification/4.3_Probabilistic_Discriminative_Models
Content: 02_4.3.3_Iterative_reweighted_least_squares
"""

import numpy as np
from scipy.linalg import solve
from typing import Tuple

class IRLSLogisticRegression:
    """
    迭代重加权最小二乘法 (IRLS) 用于逻辑回归分类器

    Parameters:
    -----------
    n_iter : int
        训练数据迭代次数 (默认值为 100)
    tol : float
        收敛阈值 (默认值为 1e-6)

    Attributes:
    -----------
    w_ : np.ndarray
        权重向量
    cost_ : list
        每次迭代中的损失值
    """
    def __init__(self, n_iter: int = 100, tol: float = 1e-6) -> None:
        self.n_iter = n_iter
        self.tol = tol
        self.w_ = None
        self.cost_ = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练逻辑回归分类器
        
        Parameters:
        -----------
        X : np.ndarray, shape = [n_samples, n_features]
            训练向量
        y : np.ndarray, shape = [n_samples]
            目标值
        """
        # 初始化权重向量
        self.w_ = np.zeros(X.shape[1] + 1)
        self.cost_ = []

        X_bias = np.insert(X, 0, 1, axis=1)  # 添加偏置项

        for _ in range(self.n_iter):
            z = np.dot(X_bias, self.w_)
            y_hat = self._sigmoid(z)
            R = np.diag(y_hat * (1 - y_hat))
            gradient = np.dot(X_bias.T, (y_hat - y))
            H = np.dot(np.dot(X_bias.T, R), X_bias)
            delta_w = solve(H, gradient)
            self.w_ -= delta_w
            cost = self._cost_function(y, y_hat)
            self.cost_.append(cost)

            # 检查收敛性
            if np.linalg.norm(delta_w) < self.tol:
                break

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
        return np.where(self._sigmoid(z) >= 0.5, 1, 0)

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        计算Sigmoid函数
        
        Parameters:
        -----------
        z : np.ndarray
            输入值
        
        Returns:
        --------
        np.ndarray
            Sigmoid函数值
        """
        return 1 / (1 + np.exp(-z))

    def _cost_function(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """
        计算交叉熵损失函数
        
        Parameters:
        -----------
        y : np.ndarray
            真实值
        y_hat : np.ndarray
            预测值
        
        Returns:
        --------
        float
            交叉熵损失值
        """
        return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

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
    主函数，运行逻辑回归并打印结果
    """
    X, y = generate_data()
    irls = IRLSLogisticRegression(n_iter=100, tol=1e-6)
    irls.fit(X, y)
    predictions = irls.predict(X)
    
    print("权重向量 w:")
    print(irls.w_)
    print("每次迭代的损失值:")
    print(irls.cost_)

if __name__ == "__main__":
    main()
