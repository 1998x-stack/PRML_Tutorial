# 04_3.1.5_Multiple_outputs

"""
Lecture: 3_Linear_Models_for_Regression/3.1_Linear_Basis_Function_Models
Content: 04_3.1.5_Multiple_outputs
"""

import numpy as np
from numpy.linalg import inv

class MultipleOutputRegression:
    def __init__(self, basis_functions: int):
        """
        初始化多输出回归模型。

        参数:
        basis_functions (int): 基函数的数量。
        """
        self.basis_functions = basis_functions
        self.W = None

    def fit(self, X: np.ndarray, T: np.ndarray):
        """
        训练多输出回归模型。

        参数:
        X (np.ndarray): 输入矩阵，形状为 (N, D)。
        T (np.ndarray): 目标矩阵，形状为 (N, K)。
        """
        N, D = X.shape
        N, K = T.shape

        # 扩展输入矩阵以包含基函数
        Phi = self._design_matrix(X)

        # 计算伪逆矩阵
        Phi_T_Phi_inv = inv(Phi.T @ Phi)
        self.W = Phi_T_Phi_inv @ Phi.T @ T

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用训练好的模型进行预测。

        参数:
        X (np.ndarray): 输入矩阵，形状为 (N, D)。

        返回:
        np.ndarray: 预测结果，形状为 (N, K)。
        """
        Phi = self._design_matrix(X)
        return Phi @ self.W

    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        生成设计矩阵，包含基函数。

        参数:
        X (np.ndarray): 输入矩阵，形状为 (N, D)。

        返回:
        np.ndarray: 设计矩阵，形状为 (N, M)。
        """
        N, D = X.shape
        Phi = np.ones((N, self.basis_functions))
        for i in range(1, self.basis_functions):
            Phi[:, i] = X[:, 0] ** i  # 示例：多项式基函数
        return Phi

# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    X = np.random.rand(100, 1)
    T = np.hstack((X ** 2, X ** 3))

    # 初始化并训练模型
    model = MultipleOutputRegression(basis_functions=3)
    model.fit(X, T)

    # 进行预测
    predictions = model.predict(X)
    print("Predictions:", predictions)