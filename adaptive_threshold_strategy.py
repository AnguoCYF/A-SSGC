import numpy as np

def adaptive_threshold_sparsify(A, base_threshold=0.5, degree_factor=0.05):

    A = np.asarray(A)
    n = A.shape[0]
    degrees = np.squeeze(np.sum(A > 0, axis=1))  # 计算每个节点的度数

    # 归一化节点度数
    max_degree = np.max(degrees)
    min_degree = np.min(degrees)
    if max_degree == min_degree:
        normalized_degrees = np.zeros_like(degrees)
    else:
        normalized_degrees = (degrees - min_degree) / (max_degree - min_degree)

    # 计算每个节点的自适应阈值
    adaptive_thresholds = base_threshold + degree_factor * (normalized_degrees[:, None] + normalized_degrees[None, :]) / 2

    # 确保阈值在 [0, 1] 范围内
    adaptive_thresholds = np.clip(adaptive_thresholds, 0, 1)

    # 稀疏化邻接矩阵
    sparse_A = np.where(A > adaptive_thresholds, A, 0)

    np.fill_diagonal(sparse_A, 1)  # 添加自环
    return sparse_A

#%%
# 示例邻接矩阵
# A = np.array([[0, 0.8, 0.2, 0.1, 0.3],
#               [0.8, 0, 0.5, 0.3, 0.4],
#               [0.2, 0.5, 0, 0.9, 0.1],
#               [0.1, 0.3, 0.9, 0, 0.5],
#               [0.3, 0.4, 0.1, 0.5, 0]])
#
#
# sparse_A_optimized = adaptive_threshold_sparsify(A, base_threshold=0.1, degree_factor=0.1)
# print("Sparse A :\n", sparse_A_optimized)
