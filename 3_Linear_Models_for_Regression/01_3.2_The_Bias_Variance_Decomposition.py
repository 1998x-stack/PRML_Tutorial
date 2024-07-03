# 01_3.2_The_Bias-Variance_Decomposition

"""
Lecture: /3_Linear_Models_for_Regression
Content: 01_3.2_The_Bias-Variance_Decomposition
"""

import numpy as np

def bias_variance_decomposition(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    计算偏差和方差

    参数:
        y_true (np.ndarray): 真实值
        y_pred (np.ndarray): 预测值

    返回:
        tuple: 偏差平方和方差
    """
    # 计算期望预测值
    y_pred_mean = np.mean(y_pred, axis=0)
    
    # 计算偏差平方
    bias_squared = np.mean((y_pred_mean - y_true) ** 2)
    
    # 计算方差
    variance = np.mean(np.var(y_pred, axis=0))
    
    return bias_squared, variance

# 示例数据
y_true = np.sin(2 * np.pi * np.array([0.1, 0.4, 0.7, 1.0]))
y_pred = np.array([
    [0.9, 1.8, 2.7, 3.5],
    [1.0, 2.0, 3.0, 4.0],
    [1.1, 2.1, 3.1, 4.2]
])

bias_squared, variance = bias_variance_decomposition(y_true, y_pred)
print(f"偏差平方: {bias_squared}")
print(f"方差: {variance}")
