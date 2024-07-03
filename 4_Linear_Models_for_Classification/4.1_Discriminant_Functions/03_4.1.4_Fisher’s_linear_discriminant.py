# 03_4.1.4_Fisher’s_linear_discriminant

"""
Lecture: 4_Linear_Models_for_Classification/4.1_Discriminant_Functions
Content: 03_4.1.4_Fisher’s_linear_discriminant
"""

import numpy as np
from typing import Tuple

class FisherLDA:
    """
    Fisher 线性判别分析 (LDA) 分类器

    Parameters:
    -----------
    n_components : int
        需要保留的线性判别维度数量

    Attributes:
    -----------
    W_ : np.ndarray
        投影矩阵
    """
    def __init__(self, n_components: int = 1) -> None:
        self.n_components = n_components
        self.W_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练LDA分类器

        Parameters:
        -----------
        X : np.ndarray, shape = [n_samples, n_features]
            训练向量
        y : np.ndarray, shape = [n_samples]
            目标值
        """
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # 计算类内散布矩阵
        S_W = np.zeros((n_features, n_features))
        mean_vectors = []
        for label in class_labels:
            X_c = X[y == label]
            mean_vec = np.mean(X_c, axis=0)
            mean_vectors.append(mean_vec)
            S_W += np.cov(X_c, rowvar=False) * (X_c.shape[0] - 1)

        # 计算总均值
        mean_overall = np.mean(X, axis=0)

        # 计算类间散布矩阵
        S_B = np.zeros((n_features, n_features))
        for i, mean_vec in enumerate(mean_vectors):
            n_c = X[y == class_labels[i]].shape[0]
            mean_diff = (mean_vec - mean_overall).reshape(n_features, 1)
            S_B += n_c * (mean_diff @ mean_diff.T)

        # 解决特征值问题
        A = np.linalg.inv(S_W) @ S_B
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        # 选择前k个最大的特征值对应的特征向量
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.W_ = eigenvectors[:, sorted_indices[:self.n_components]].real

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        将输入数据投影到LDA新空间

        Parameters:
        -----------
        X : np.ndarray, shape = [n_samples, n_features]
            输入向量

        Returns:
        --------
        np.ndarray
            投影后的数据
        """
        return X @ self.W_

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
    主函数，运行LDA并打印结果
    """
    X, y = generate_data()
    lda = FisherLDA(n_components=1)
    lda.fit(X, y)
    X_projected = lda.transform(X)
    
    print("投影矩阵 W:")
    print(lda.W_)
    print("投影后的数据形状:", X_projected.shape)

if __name__ == "__main__":
    main()
