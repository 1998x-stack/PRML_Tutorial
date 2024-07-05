# 02_7.1.3_Multiclass_SVMs

"""
Lecture: 7_Sparse_Kernel_Machines/7.1_Maximum_Margin_Classifiers
Content: 02_7.1.3_Multiclass_SVMs
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List, Callable


class MulticlassSVM:
    """
    多分类支持向量机（Multiclass SVM）实现类

    Attributes:
        C (float): 正则化参数
        kernel (Callable[[np.ndarray, np.ndarray], float]): 核函数
        classifiers (List[Tuple[int, int, np.ndarray]]): 一对一分类器的列表，包含类别对和权重向量
    """

    def __init__(self, C: float = 1.0, kernel: Callable[[np.ndarray, np.ndarray], float] = None):
        self.C = C
        self.kernel = kernel if kernel else self.linear_kernel
        self.classifiers = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        拟合模型，训练多分类支持向量机

        Args:
            X (np.ndarray): 训练数据特征，形状为 (n_samples, n_features)
            y (np.ndarray): 训练数据标签，形状为 (n_samples,)
        """
        self.classifiers = []
        classes = np.unique(y)
        for i, class_i in enumerate(classes):
            for j, class_j in enumerate(classes):
                if i < j:
                    # 提取类别 i 和类别 j 的数据
                    idx = np.where((y == class_i) | (y == class_j))
                    X_ij = X[idx]
                    y_ij = y[idx]
                    y_ij = np.where(y_ij == class_i, 1, -1)

                    # 训练二分类SVM
                    weights = self._fit_binary_svm(X_ij, y_ij)
                    self.classifiers.append((class_i, class_j, weights))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测新数据的类别

        Args:
            X (np.ndarray): 测试数据特征，形状为 (n_samples, n_features)

        Returns:
            np.ndarray: 预测标签，形状为 (n_samples,)
        """
        votes = np.zeros((X.shape[0], len(self.classifiers)))

        for k, (class_i, class_j, weights) in enumerate(self.classifiers):
            predictions = np.sign(X.dot(weights[:-1]) + weights[-1])
            votes[:, k] = np.where(predictions == 1, class_i, class_j)

        # 投票确定最终类别
        y_pred = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=votes)
        return y_pred

    def _fit_binary_svm(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        训练二分类SVM

        Args:
            X (np.ndarray): 二分类数据特征
            y (np.ndarray): 二分类数据标签

        Returns:
            np.ndarray: 学习到的权重向量
        """
        n_samples, n_features = X.shape
        K = self.kernel(X, X)

        # 定义优化问题
        P = np.outer(y, y) * K
        q = -np.ones(n_samples)
        G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
        h = np.hstack((np.zeros(n_samples), self.C * np.ones(n_samples)))
        A = y.reshape(1, -1)
        b = np.zeros(1)

        def objective(alpha: np.ndarray) -> float:
            return 0.5 * alpha.dot(P).dot(alpha) - alpha.sum()

        def zerofun(alpha: np.ndarray) -> float:
            return alpha.dot(y)

        # 求解拉格朗日乘数
        constraints = {'type': 'eq', 'fun': zerofun}
        bounds = [(0, self.C) for _ in range(n_samples)]
        result = minimize(objective, np.zeros(n_samples), bounds=bounds, constraints=constraints)
        alpha = result.x

        # 计算权重向量
        support_vectors = alpha > 1e-5
        alpha = alpha[support_vectors]
        support_vectors_X = X[support_vectors]
        support_vectors_y = y[support_vectors]

        weights = np.sum(alpha * support_vectors_y[:, np.newaxis] * support_vectors_X, axis=0)
        bias = np.mean(support_vectors_y - support_vectors_X.dot(weights))

        return np.append(weights, bias)

    @staticmethod
    def linear_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
        """
        线性核函数

        Args:
            x1 (np.ndarray): 输入向量1
            x2 (np.ndarray): 输入向量2

        Returns:
            float: 线性核的计算结果
        """
        return np.dot(x1, x2.T)


def main():
    # 示例数据
    X = np.array([
        [2, 3],
        [3, 3],
        [3, 4],
        [5, 6],
        [6, 6],
        [6, 5],
        [10, 10],
        [10, 11],
        [11, 11]
    ])
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    # 初始化和训练模型
    svm = MulticlassSVM(C=1.0)
    svm.fit(X, y)

    # 测试数据
    X_test = np.array([
        [4, 5],
        [8, 8],
        [10, 12]
    ])

    # 预测
    predictions = svm.predict(X_test)
    print("Predicted labels:", predictions)


if __name__ == "__main__":
    main()