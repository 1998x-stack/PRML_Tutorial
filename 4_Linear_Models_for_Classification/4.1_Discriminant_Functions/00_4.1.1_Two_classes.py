# 00_4.1.1_Two_classes

"""
Lecture: 4_Linear_Models_for_Classification/4.1_Discriminant_Functions
Content: 00_4.1.1_Two_classes
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple

class LinearDiscriminantAnalysis:
    """线性判别分析（Linear Discriminant Analysis, LDA）用于两类分类问题的类。
    
    该类实现了线性判别函数 y(x) = w^T x + w_0，其中 w 为权重向量，w_0 为偏置。
    """
    
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """拟合LDA模型。
        
        参数:
        X (np.ndarray): 输入数据，形状为 (n_samples, n_features)
        y (np.ndarray): 标签数据，形状为 (n_samples,)
        """
        n_samples, n_features = X.shape
        X_with_bias = np.hstack([np.ones((n_samples, 1)), X])
        
        # 定义目标函数
        def objective(w):
            predictions = X_with_bias @ w
            return np.sum((predictions - y) ** 2)
        
        # 初始权重
        initial_weights = np.zeros(n_features + 1)
        
        # 最小化目标函数
        result = minimize(objective, initial_weights)
        if not result.success:
            raise ValueError("优化失败")
        
        self.weights = result.x[1:]
        self.bias = result.x[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测新数据的类别。
        
        参数:
        X (np.ndarray): 输入数据，形状为 (n_samples, n_features)
        
        返回:
        np.ndarray: 预测的类别，形状为 (n_samples,)
        """
        return np.sign(X @ self.weights + self.bias)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """计算判别函数的值。
        
        参数:
        X (np.ndarray): 输入数据，形状为 (n_samples, n_features)
        
        返回:
        np.ndarray: 判别函数的值，形状为 (n_samples,)
        """
        return X @ self.weights + self.bias

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算模型的准确率。
        
        参数:
        X (np.ndarray): 输入数据，形状为 (n_samples, n_features)
        y (np.ndarray): 标签数据，形状为 (n_samples,)
        
        返回:
        float: 准确率
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

# 数据生成和模型测试
def generate_data(n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """生成两类分类问题的模拟数据。
    
    参数:
    n_samples (int): 样本数量
    
    返回:
    Tuple[np.ndarray, np.ndarray]: 输入数据和标签
    """
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)
    return X, y

def main():
    """主函数，用于测试LDA模型。
    """
    X, y = generate_data(200)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    accuracy = lda.score(X, y)
    print(f"模型的准确率为: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
