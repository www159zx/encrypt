import numpy as np
import pandas as pd

# 读取CSV文件
#df = pd.read_csv('chen_max.csv', header=None)  # 如果CSV没有表头，添加header=None
df = pd.read_csv('output.csv', header=None)
# 将DataFrame转换为NumPy数组
data = df.values.flatten()

def calculate_shannon_entropy(data, base=2):
    # 计算概率分布
    probabilities = np.unique(data, return_counts=True)[1] / len(data)

    # 计算香农熵
    entropy = -np.sum(probabilities[probabilities > 0] * np.log(base, probabilities[probabilities > 0]))

    return entropy

# 计算香农熵
entropy = calculate_shannon_entropy(data)
print(f"Shannon Entropy of the array: {entropy}")