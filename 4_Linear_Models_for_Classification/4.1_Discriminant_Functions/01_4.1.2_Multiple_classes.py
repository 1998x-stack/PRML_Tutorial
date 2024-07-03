# 01_4.1.2_Multiple_classes

"""
Lecture: 4_Linear_Models_for_Classification/4.1_Discriminant_Functions
Content: 01_4.1.2_Multiple_classes
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List

class MultiClassLDA:
    """多类判别分析（Multi-class Linear Discriminant Analysis, LDA）用于多类分类问题的类。
    
    该类实现了多类线性判别函数 y_k(x) = w_k^T x + w_k0，其中 w_k 为类别 k 的权重向量，w_k0 为偏置。
    """
    
    def __init__(self):
        self.weights = None
        self.biases = None
        self.classes = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """拟合LDA模型。
        
        参数:
        X (np.ndarray): 输入数据，形状为 (n_samples, n_features)
        y (np.ndarray): 标签数据，形状为 (n_samples,)
        """
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes)
        
        X_with_bias = np.hstack([np.ones((n_samples, 1)), X])
        
        def objective(W):
            W = W.reshape((n_classes, n_features + 1))
            scores = X_with_bias @ W.T
            probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
            loss = -np.sum(np.log(probs[np.arange(n_samples), y]))
            return loss
        
        initial_weights = np.zeros((n_classes, n_features + 1)).flatten()
        result = minimize(objective, initial_weights, method='L-BFGS-B')
        
        if not result.success:
            raise ValueError("优化失败")
        
        W_optimal = result.x.reshape((n_classes, n_features + 1))
        self.biases = W_optimal[:, 0]
        self.weights = W_optimal[:, 1:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测新数据的类别。
        
        参数:
        X (np.ndarray): 输入数据，形状为 (n_samples, n_features)
        
        返回:
        np.ndarray: 预测的类别，形状为 (n_samples,)
        """
        scores = X @ self.weights.T + self.biases
        return self.classes[np.argmax(scores, axis=1)]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """计算判别函数的值。
        
        参数:
        X (np.ndarray): 输入数据，形状为 (n_samples, n_features)
        
        返回:
        np.ndarray: 判别函数的值，形状为 (n_samples, n_classes)
        """
        return X @ self.weights.T + self.biases

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
    """主函数，用于测试多类LDA模型。
    """
    X, y = generate_data(200, 3)
    lda = MultiClassLDA()
    lda.fit(X, y)
    accuracy = lda.score(X, y)
    print(f"模型的准确率为: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
