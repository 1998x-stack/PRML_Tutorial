# 04_4.1.5_Relation_to_least_squares

"""
Lecture: 4_Linear_Models_for_Classification/4.1_Discriminant_Functions
Content: 04_4.1.5_Relation_to_least_squares
"""

import numpy as np
from typing import Tuple

class LeastSquaresFisherLDA:
    """
    最小二乘法与Fisher线性判别分析 (LDA) 分类器
    
    Parameters:
    -----------
    None
    
    Attributes:
    -----------
    w_ : np.ndarray
        权重向量
    w0_ : float
        偏置
    """
    def __init__(self) -> None:
        self.w_ = None
        self.w0_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练分类器
        
        Parameters:
        -----------
        X : np.ndarray, shape = [n_samples, n_features]
            训练向量
        y : np.ndarray, shape = [n_samples]
            目标值
        """
        # 计算总均值
        mean_overall = np.mean(X, axis=0)
        
        # 计算类均值
        mean_vectors = []
        class_labels = np.unique(y)
        for label in class_labels:
            mean_vectors.append(np.mean(X[y == label], axis=0))
        
        # 计算类内散布矩阵
        S_W = np.zeros((X.shape[1], X.shape[1]))
        for label, mv in zip(class_labels, mean_vectors):
            class_scatter = np.zeros((X.shape[1], X.shape[1]))
            for row in X[y == label]:
                row, mv = row.reshape(X.shape[1], 1), mv.reshape(X.shape[1], 1)
                class_scatter += (row - mv) @ (row - mv).T
            S_W += class_scatter

        # 计算类间散布矩阵
        S_B = np.zeros((X.shape[1], X.shape[1]))
        for i, mean_vec in enumerate(mean_vectors):
            n = X[y == class_labels[i], :].shape[0]
            mean_vec = mean_vec.reshape(X.shape[1], 1)
            overall_mean = mean_overall.reshape(X.shape[1], 1)
            S_B += n * (mean_vec - overall_mean) @ (mean_vec - overall_mean).T

        # 计算权重向量
        A = np.linalg.inv(S_W) @ (mean_vectors[1] - mean_vectors[0])
        self.w_ = A

        # 计算偏置
        N1 = X[y == class_labels[0]].shape[0]
        N2 = X[y == class_labels[1]].shape[0]
        self.w0_ = -self.w_.T @ mean_overall

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测输入数据的类别
        
        Parameters:
        -----------
        X : np.ndarray, shape = [n_samples, n_features]
            输入向量
        
        Returns:
        --------
        np.ndarray
            类别预测
        """
        return np.where((X @ self.w_ + self.w0_) >= 0, 1, 0)

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
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))
    return X, y

def main() -> None:
    """
    主函数，运行分类器并打印结果
    """
    X, y = generate_data()
    classifier = LeastSquaresFisherLDA()
    classifier.fit(X, y)
    predictions = classifier.predict(X)
    
    print("权重向量 w:")
    print(classifier.w_)
    print("偏置 w0:")
    print(classifier.w0_)
    print("预测值:")
    print(predictions)

if __name__ == "__main__":
    main()
