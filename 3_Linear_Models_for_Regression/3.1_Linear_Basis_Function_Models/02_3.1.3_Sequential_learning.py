# 02_3.1.3_Sequential_learning

"""
Lecture: 3_Linear_Models_for_Regression/3.1_Linear_Basis_Function_Models
Content: 02_3.1.3_Sequential_learning
"""

import numpy as np

class SequentialLearning:
    """
    实现顺序学习算法类
    Attributes:
        learning_rate (float): 学习率
        weights (np.ndarray): 模型参数向量
    """

    def __init__(self, learning_rate: float, n_features: int):
        """
        初始化顺序学习算法类
        Args:
            learning_rate (float): 学习率
            n_features (int): 特征数量
        """
        self.learning_rate = learning_rate
        self.weights = np.zeros(n_features)

    def update_weights(self, x: np.ndarray, t: float) -> None:
        """
        更新模型参数
        Args:
            x (np.ndarray): 输入特征向量
            t (float): 目标值
        """
        prediction = np.dot(self.weights, x)
        error = t - prediction
        self.weights += self.learning_rate * error * x

    def train(self, X: np.ndarray, T: np.ndarray) -> None:
        """
        训练模型
        Args:
            X (np.ndarray): 输入特征矩阵
            T (np.ndarray): 目标值向量
        """
        for x, t in zip(X, T):
            self.update_weights(x, t)

    def predict(self, x: np.ndarray) -> float:
        """
        预测目标值
        Args:
            x (np.ndarray): 输入特征向量
        Returns:
            float: 预测值
        """
        return np.dot(self.weights, x)

    def evaluate(self, X: np.ndarray, T: np.ndarray) -> float:
        """
        评估模型性能
        Args:
            X (np.ndarray): 输入特征矩阵
            T (np.ndarray): 目标值向量
        Returns:
            float: 平均平方误差
        """
        predictions = np.dot(X, self.weights)
        errors = T - predictions
        return np.mean(errors ** 2)

if __name__ == "__main__":
    # 示例数据
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    T_train = np.array([2.5, 3.5, 4.5, 5.5])

    model = SequentialLearning(learning_rate=0.01, n_features=2)
    model.train(X_train, T_train)

    X_test = np.array([[5, 6]])
    prediction = model.predict(X_test[0])
    print(f"Prediction: {prediction}")

    mse = model.evaluate(X_train, T_train)
    print(f"Mean Squared Error: {mse}")