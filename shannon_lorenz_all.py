import numpy as np
import pandas as pd

# 读取CSV文件
df = pd.read_csv('lorenz_data.csv')

# 将x, y, z值组合成一个三维坐标数组
coordinates_3d = df[['x', 'y', 'z']].values

# 将三维坐标展平为一维数组
data = coordinates_3d.flatten()

def calculate_shannon_entropy_3d_flat(data, num_bins=100):
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

# 计算三维坐标展平后的香农熵
entropy = calculate_shannon_entropy_3d_flat(data)           #num_bins=100   6.332665020586979
print(f"Shannon Entropy for flattened 3D coordinates: {entropy}")    #num_bins=10  3.042639561358319