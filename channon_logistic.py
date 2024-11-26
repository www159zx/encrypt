import numpy as np
import pandas as pd





def calculate_shannon_entropy(num_bins=10):
    # 将坐标转换为一维数组
    data = pd.read_csv('logustic_data.csv', header=None).values.flatten()

    # 确定每个维度的最小值和最大值
    min_val = np.min(data)
    max_val = np.max(data)

    # 将数据归一化到[0, 1]区间
    data_normalized = (data - min_val) / (max_val - min_val)

    # 将归一化的数据分配到不同的区间
    hist, bin_edges = np.histogram(data_normalized, bins=num_bins, density=True)

    # 计算概率分布
    probabilities = hist / hist.sum()

    # 计算香农熵
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

    return entropy



entropy = calculate_shannon_entropy()  #num_bins=100   6.394958397781797
print(f"The Shannon Entropy is: {entropy}")     #num_bins=10    3.1758045054006763
