# 03_4.3.4_Multiclass_logistic_regression

"""
Lecture: 4_Linear_Models_for_Classification/4.3_Probabilistic_Discriminative_Models
Content: 03_4.3.4_Multiclass_logistic_regression
"""

import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
from typing import Tuple

class MulticlassLogisticRegression:
    """
    多类别逻辑回归分类器

    Parameters:
    -----------
    n_iter : int
        训练数据迭代次数 (默认值为 100)
    tol : float
        收敛阈值 (默认值为 1e-6)

    Attributes:
    -----------
    W_ : np.ndarray
        权重矩阵
    cost_ : list
        每次迭代中的损失值
    """
    def __init__(self, n_iter: int = 100, tol: float = 1e-6) -> None:
        self.n_iter = n_iter
        self.tol = tol
        self.W_ = None
        self.cost_ = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练多类别逻辑回归分类器
        
        Parameters:
        -----------
        X : np.ndarray, shape = [n_samples, n_features]
            训练向量
        y : np.ndarray, shape = [n_samples]
            目标值
        """
        n_samples, n_features = X.shape
        n_classes = np.unique(y).size
        self.W_ = np.zeros((n_features + 1, n_classes))

        X_bias = np.insert(X, 0, 1, axis=1)  # 添加偏置项
        y_one_hot = np.eye(n_classes)[y]  # 转换为one-hot编码

        for _ in range(self.n_iter):
            z = np.dot(X_bias, self.W_)
            y_hat = softmax(z, axis=1)
            gradient = np.dot(X_bias.T, (y_hat - y_one_hot)) / n_samples
            H = np.dot(X_bias.T, X_bias * y_hat * (1 - y_hat)[:, np.newaxis]) / n_samples
            delta_W = np.linalg.solve(H, gradient)
            self.W_ -= delta_W
            cost = self._cost_function(y_one_hot, y_hat)
            self.cost_.append(cost)

            # 检查收敛性
            if np.linalg.norm(delta_W) < self.tol:
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
        z = np.dot(X_bias, self.W_)
        y_hat = softmax(z, axis=1)
        return np.argmax(y_hat, axis=1)

    def _cost_function(self, y_true: np.ndarray, y_hat: np.ndarray) -> float:
        """
        计算交叉熵损失函数
        
        Parameters:
        -----------
        y_true : np.ndarray
            真实值
        y_hat : np.ndarray
            预测值
        
        Returns:
        --------
        float
            交叉熵损失值
        """
        return -np.mean(np.sum(y_true * np.log(y_hat), axis=1))

def generate_data(n_samples: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成多类别数据集
    
    Parameters:
    -----------
    n_samples : int
        样本数量 (默认值为 300)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        特征矩阵和目标值向量
    """
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3])
    return X, y

def main() -> None:
    """
    主函数，运行多类别逻辑回归并打印结果
    """
    X, y = generate_data()
    clf = MulticlassLogisticRegression(n_iter=100, tol=1e-6)
    clf.fit(X, y)
    predictions = clf.predict(X)
    
    print("权重矩阵 W:")
    print(clf.W_)
    print("每次迭代的损失值:")
    print(clf.cost_)

if __name__ == "__main__":
    main()
