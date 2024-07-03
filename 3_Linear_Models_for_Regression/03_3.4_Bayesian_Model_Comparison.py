# 03_3.4_Bayesian_Model_Comparison

"""
Lecture: /3_Linear_Models_for_Regression
Content: 03_3.4_Bayesian_Model_Comparison
"""

import numpy as np
from scipy.linalg import det, inv
from scipy.special import logsumexp

class BayesianModelComparison:
    """
    贝叶斯模型比较类

    参数:
        models (list): 模型列表，每个模型是一个包含 'prior' 和 'likelihood' 函数的字典
    """
    def __init__(self, models: list):
        self.models = models
    
    def compute_evidence(self, model, X: np.ndarray, t: np.ndarray) -> float:
        """
        计算模型证据

        参数:
            model (dict): 包含 'prior' 和 'likelihood' 函数的字典
            X (np.ndarray): 输入数据
            t (np.ndarray): 目标值

        返回:
            float: 模型证据的对数值
        """
        prior = model['prior']
        likelihood = model['likelihood']
        w_map = self._find_map(likelihood, prior, X, t)
        hessian = self._compute_hessian(likelihood, prior, w_map, X, t)
        log_evidence = (likelihood(X, t, w_map) +
                        prior(w_map) +
                        0.5 * np.log(det(hessian)) -
                        0.5 * w_map.size * np.log(2 * np.pi))
        return log_evidence
    
    def compare_models(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        比较模型，计算每个模型的证据和贝叶斯因子

        参数:
            X (np.ndarray): 输入数据
            t (np.ndarray): 目标值

        返回:
            np.ndarray: 模型的对数证据和贝叶斯因子
        """
        log_evidences = [self.compute_evidence(model, X, t) for model in self.models]
        log_bayes_factors = log_evidences - logsumexp(log_evidences)
        return np.exp(log_bayes_factors)
    
    def _find_map(self, likelihood, prior, X, t):
        # 通过最大化后验找到MAP估计
        # 这里使用一个简单的梯度下降示例，实际应用中可以使用更复杂的优化方法
        w_map = np.zeros(X.shape[1])
        learning_rate = 0.01
        for _ in range(100):
            grad = self._compute_gradient(likelihood, prior, w_map, X, t)
            w_map += learning_rate * grad
        return w_map
    
    def _compute_gradient(self, likelihood, prior, w, X, t):
        # 计算梯度
        return likelihood.gradient(X, t, w) + prior.gradient(w)
    
    def _compute_hessian(self, likelihood, prior, w, X, t):
        # 计算Hessian矩阵
        return likelihood.hessian(X, t, w) + prior.hessian(w)

# 示例使用
if __name__ == "__main__":
    # 定义模型的先验和似然函数
    def prior(w):
        return -0.5 * np.sum(w**2)

    def likelihood(X, t, w):
        y = X @ w
        return -0.5 * np.sum((t - y)**2)
    
    model1 = {'prior': prior, 'likelihood': likelihood}
    model2 = {'prior': prior, 'likelihood': likelihood}
    
    # 模拟数据
    X_train = np.random.randn(100, 2)
    t_train = X_train @ np.array([1.5, -2.0]) + np.random.randn(100)
    
    # 进行模型比较
    comparison = BayesianModelComparison(models=[model1, model2])
    log_evidences = comparison.compare_models(X_train, t_train)
    
    print("模型的对数证据: ", log_evidences)
