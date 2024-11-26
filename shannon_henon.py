import pandas as pd
import numpy as np


# 读取CSV文件
x_data = pd.read_csv('henon_x_data.csv', header=None).values.flatten()
y_data = pd.read_csv('henon_y_data.csv', header=None).values.flatten()



# 将x和y值组合成一个二维坐标数组
coordinates = np.column_stack((x_data, y_data))


def calculate_shannon_entropy(coordinates, num_bins=100):
    # 将坐标转换为一维数组
    data = coordinates.flatten()

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



entropy = calculate_shannon_entropy(coordinates)  #num_bins=100   6.189566638970422
print(f"The Shannon Entropy is: {entropy}")     #num_bins=10    2.9890860617212924


