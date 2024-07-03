# 06_4.1.7_The_perceptron_algorithm

"""
Lecture: 4_Linear_Models_for_Classification/4.1_Discriminant_Functions
Content: 06_4.1.7_The_perceptron_algorithm
"""

import numpy as np
from typing import Tuple

class Perceptron:
    """
    Perceptron 分类器
    
    Parameters:
    -----------
    learning_rate : float
        学习率 (默认值为 1.0)
    n_iter : int
        训练数据迭代次数 (默认值为 1000)
        
    Attributes:
    -----------
    w_ : np.ndarray
        权重向量
    errors_ : list
        每次迭代中的误分类数
    """
    def __init__(self, learning_rate: float = 1.0, n_iter: int = 1000) -> None:
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.w_ = None
        self.errors_ = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练感知器分类器
        
        Parameters:
        -----------
        X : np.ndarray, shape = [n_samples, n_features]
            训练向量
        y : np.ndarray, shape = [n_samples]
            目标值
        """
        # 初始化权重向量，包含偏置项
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            if errors == 0:
                break

    def net_input(self, X: np.ndarray) -> np.ndarray:
        """
        计算净输入
        
        Parameters:
        -----------
        X : np.ndarray
            输入向量
        
        Returns:
        --------
        np.ndarray
            净输入值
        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X: np.ndarray) -> int:
        """
        返回类标预测值
        
        Parameters:
        -----------
        X : np.ndarray
            输入向量
        
        Returns:
        --------
        int
            类标预测值
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)

def generate_data(n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成二分类数据集
    
    Parameters:
    -----------
    n_samples : int
        样本数 (默认值为 100)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        特征矩阵和目标值向量
    """
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)
    return X, y

def main() -> None:
    """
    主函数，运行感知器算法并打印结果
    """
    X, y = generate_data()
    perceptron = Perceptron(learning_rate=1.0, n_iter=1000)
    perceptron.fit(X, y)
    
    print("训练后的权重向量:", perceptron.w_)
    print("每次迭代中的误分类数:", perceptron.errors_)

if __name__ == "__main__":
    main()
