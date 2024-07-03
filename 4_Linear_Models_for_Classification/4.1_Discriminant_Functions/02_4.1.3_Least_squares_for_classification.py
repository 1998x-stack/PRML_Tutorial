# 02_4.1.3_Least_squares_for_classification

"""
Lecture: 4_Linear_Models_for_Classification/4.1_Discriminant_Functions
Content: 02_4.1.3_Least_squares_for_classification
"""

import numpy as np
from numpy.linalg import pinv
from typing import Tuple

class LeastSquaresClassifier:
    """最小二乘分类器用于多类分类问题的类。
    
    该类实现了通过最小化平方误差来拟合线性判别函数 y_k(x) = w_k^T x + w_k0。
    """
    
    def __init__(self):
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """拟合最小二乘分类器。
        
        参数:
        X (np.ndarray): 输入数据，形状为 (n_samples, n_features)
        y (np.ndarray): 标签数据，形状为 (n_samples,)
        """
        n_samples, n_features = X.shape
        X_with_bias = np.hstack([np.ones((n_samples, 1)), X])
        n_classes = len(np.unique(y))
        
        # 将标签转换为 1-of-K 编码
        T = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            T[i, y[i]] = 1
        
        # 计算权重矩阵
        self.weights = pinv(X_with_bias).dot(T)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测新数据的类别。
        
        参数:
        X (np.ndarray): 输入数据，形状为 (n_samples, n_features)
        
        返回:
        np.ndarray: 预测的类别，形状为 (n_samples,)
        """
        X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        scores = X_with_bias.dot(self.weights)
        return np.argmax(scores, axis=1)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """计算判别函数的值。
        
        参数:
        X (np.ndarray): 输入数据，形状为 (n_samples, n_features)
        
        返回:
        np.ndarray: 判别函数的值，形状为 (n_samples, n_classes)
        """
        X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_with_bias.dot(self.weights)
    
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
def generate_data(n_samples: int = 100, n_classes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """生成多类分类问题的模拟数据。
    
    参数:
    n_samples (int): 样本数量
    n_classes (int): 类别数量
    
    返回:
    Tuple[np.ndarray, np.ndarray]: 输入数据和标签
    """
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = np.random.randint(n_classes, size=n_samples)
    return X, y

def main():
    """主函数，用于测试最小二乘分类器。
    """
    X, y = generate_data(200, 3)
    lsc = LeastSquaresClassifier()
    lsc.fit(X, y)
    accuracy = lsc.score(X, y)
    print(f"模型的准确率为: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
