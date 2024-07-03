# 01_2.5.2_Nearest-neighbour_methods

"""
Lecture: 2_Probability_Distributions/2.5_Nonparametric_Methods
Content: 01_2.5.2_Nearest-neighbour_methods
"""

import numpy as np
from scipy.spatial import distance
from typing import List, Tuple

class NearestNeighbour:
    def __init__(self, k: int = 1):
        """
        初始化最近邻类
        
        参数:
        k (int): 最近邻的数量
        """
        self.k = k

    def fit(self, data: np.ndarray, labels: np.ndarray) -> None:
        """
        拟合最近邻分类器
        
        参数:
        data (np.ndarray): 训练数据集
        labels (np.ndarray): 训练数据标签
        """
        self.data = data
        self.labels = labels

    def _find_neighbours(self, point: np.ndarray) -> List[int]:
        """
        找到给定点的最近邻
        
        参数:
        point (np.ndarray): 给定点
        
        返回:
        List[int]: 最近邻的索引列表
        """
        distances = distance.cdist([point], self.data, metric='euclidean').flatten()
        neighbour_indices = np.argsort(distances)[:self.k]
        return neighbour_indices

    def predict(self, points: np.ndarray) -> np.ndarray:
        """
        预测给定点的标签
        
        参数:
        points (np.ndarray): 测试数据集
        
        返回:
        np.ndarray: 预测标签
        """
        predictions = []
        for point in points:
            neighbour_indices = self._find_neighbours(point)
            neighbour_labels = self.labels[neighbour_indices]
            predicted_label = np.bincount(neighbour_labels).argmax()
            predictions.append(predicted_label)
        return np.array(predictions)

    def predict_proba(self, points: np.ndarray) -> np.ndarray:
        """
        预测给定点的标签概率
        
        参数:
        points (np.ndarray): 测试数据集
        
        返回:
        np.ndarray: 预测标签的概率
        """
        probabilities = []
        for point in points:
            neighbour_indices = self._find_neighbours(point)
            neighbour_labels = self.labels[neighbour_indices]
            counts = np.bincount(neighbour_labels, minlength=np.max(self.labels) + 1)
            probabilities.append(counts / self.k)
        return np.array(probabilities)

# 示例用法
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(0)
    data = np.random.randn(100, 2)
    labels = np.random.randint(0, 2, 100)
    
    # 创建最近邻分类器
    nn = NearestNeighbour(k=3)
    nn.fit(data, labels)
    
    # 预测测试点的标签
    test_points = np.random.randn(10, 2)
    predictions = nn.predict(test_points)
    probabilities = nn.predict_proba(test_points)
    
    print("预测标签:", predictions)
    print("预测标签的概率:", probabilities)