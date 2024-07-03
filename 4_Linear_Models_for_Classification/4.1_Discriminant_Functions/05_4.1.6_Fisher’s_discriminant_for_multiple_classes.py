# 05_4.1.6_Fisher’s_discriminant_for_multiple_classes

"""
Lecture: 4_Linear_Models_for_Classification/4.1_Discriminant_Functions
Content: 05_4.1.6_Fisher’s_discriminant_for_multiple_classes
"""
import numpy as np
from scipy.linalg import eigh
from typing import Tuple

class MultiClassFLDA:
    """
    多类Fisher线性判别分析 (FLDA) 分类器

    Parameters:
    -----------
    n_components : int
        需要保留的线性判别维度数量

    Attributes:
    -----------
    W_ : np.ndarray
        投影矩阵
    """
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.W_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练FLDA分类器

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
        for label in class_labels:
            X_c = X[y == label]
            mean_vec = np.mean(X_c, axis=0)
            S_W += np.cov(X_c, rowvar=False) * (X_c.shape[0] - 1)

        # 计算总均值
        mean_overall = np.mean(X, axis=0)

        # 计算类间散布矩阵
        S_B = np.zeros((n_features, n_features))
        for label in class_labels:
            X_c = X[y == label]
            n_c = X_c.shape[0]
            mean_vec = np.mean(X_c, axis=0)
            mean_diff = (mean_vec - mean_overall).reshape(n_features, 1)
            S_B += n_c * (mean_diff @ mean_diff.T)

        # 解决特征值问题
        A = np.linalg.inv(S_W) @ S_B
        eigenvalues, eigenvectors = eigh(A)
        
        # 选择前k个最大的特征值对应的特征向量
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.W_ = eigenvectors[:, sorted_indices[:self.n_components]]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        将输入数据投影到FLDA新空间

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

def generate_data(n_samples: int = 300, n_features: int = 4, n_classes: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成多类分类数据集

    Parameters:
    -----------
    n_samples : int
        样本数量 (默认值为 300)
    n_features : int
        特征数量 (默认值为 4)
    n_classes : int
        类别数量 (默认值为 3)

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        特征矩阵和目标值向量
    """
    np.random.seed(0)
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice(range(n_classes), n_samples)
    return X, y

def main() -> None:
    """
    主函数，运行FLDA并打印结果
    """
    X, y = generate_data()
    flda = MultiClassFLDA(n_components=2)
    flda.fit(X, y)
    X_projected = flda.transform(X)
    
    print("投影矩阵 W:")
    print(flda.W_)
    print("投影后的数据形状:", X_projected.shape)

if __name__ == "__main__":
    main()