import pandas as pd
import numpy as np

# 假设你已经有了以下代码生成数据并保存为 CSV 文件
# df = pd.DataFrame({'x': save_x, 'y': save_y, 'z': save_z})
# df.to_csv('lorenz_data.csv', index=False)

def shannon_entropy(data):
    # 计算数据的频率分布
    value, counts = np.unique(data, return_counts=True)
    probabilities = counts / counts.sum()  # 计算概率分布

    # 计算 Shannon Entropy
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # 添加小常数以避免 log(0)
    return entropy

# 读取 CSV 文件
data = pd.read_csv('lorenz_data.csv')

# 假设你想计算 'x' 列的 Shannon Entropy
entropy_x = shannon_entropy(data['x'].values)  # 将 Series 转换为 NumPy 数组
print("Shannon Entropy of x:", entropy_x)

# 如果需要计算 'y' 或 'z' 列的 Shannon Entropy，可以按如下方式操作：
entropy_y = shannon_entropy(data['y'].values)
print("Shannon Entropy of y:", entropy_y)

entropy_z = shannon_entropy(data['z'].values)
print("Shannon Entropy of z:", entropy_z)


#Shannon Entropy of x: 13.135707987679448
#Shannon Entropy of y: 13.135707987679448
#Shannon Entropy of z: 13.135707987679448
#计算单个维度的熵