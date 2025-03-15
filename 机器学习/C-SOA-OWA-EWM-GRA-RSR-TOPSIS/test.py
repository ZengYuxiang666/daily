import numpy as np


def entropy_weight(matrix):
    """计算熵权法的权重"""
    matrix = np.array(matrix)
    P = matrix / matrix.sum(axis=0)
    E = -np.nansum(P * np.log(P), axis=0) / np.log(len(matrix))
    weights = (1 - E) / np.sum(1 - E)
    return weights


def topsis(matrix, weights, benefit_criteria):
    """TOPSIS 计算理想解与负理想解"""
    matrix = np.array(matrix)

    # 归一化
    norm_matrix = matrix / np.sqrt((matrix ** 2).sum(axis=0))

    # 乘以权重
    weighted_matrix = norm_matrix * weights

    # 确定正理想解与负理想解
    ideal_best = np.max(weighted_matrix, axis=0) if benefit_criteria else np.min(weighted_matrix, axis=0)
    ideal_worst = np.min(weighted_matrix, axis=0) if benefit_criteria else np.max(weighted_matrix, axis=0)

    # 计算与理想解的欧几里得距离
    dist_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))

    # 计算 TOPSIS 得分
    scores = dist_worst / (dist_best + dist_worst)
    rankings = scores.argsort()[::-1] + 1  # 排序（降序）

    return scores, rankings


# 示例数据（决策矩阵）
decision_matrix = np.array([
    [0.8, 200, 3000],
    [0.6, 180, 3200],
    [0.9, 220, 3100],
    [0.7, 210, 3050]
])

# 计算熵权
weights = entropy_weight(decision_matrix)

# TOPSIS 计算排名（假设所有指标为效益型）
scores, rankings = topsis(decision_matrix, weights, benefit_criteria=True)

print("权重:", weights)
print("TOPSIS 评分:", scores)
print("最终排名:", rankings)
