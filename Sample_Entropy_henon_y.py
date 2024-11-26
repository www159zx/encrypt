import numpy as np


def sample_entropy(data, m, r):
    def _max_distance(x, y):
        return np.max(np.abs(x - y))

    N = len(data)

    # 计算 m 长度模板
    templates = np.array([data[i:i + m] for i in range(N - m)])
    count_m = 0
    for i in range(len(templates)):
        for j in range(i + 1, len(templates)):
            if _max_distance(templates[i], templates[j]) <= r:
                count_m += 1
                print(f"Count_m pair: {templates[i]}, {templates[j]}")  # 检测

    # 计算 m+1 长度模板
    templates = np.array([data[i:i + m + 1] for i in range(N - m - 1)])
    count_m1 = 0
    for i in range(len(templates)):
        for j in range(i + 1, len(templates)):
            if _max_distance(templates[i], templates[j]) <= r:
                count_m1 += 1
                #print(f"Count_m1 pair: {templates[i]}, {templates[j]}")  # 检测

    return -np.log(count_m1 / count_m) if count_m > 0 else np.inf

# 读取 .csv 文件
loaded_data = np.loadtxt('henon_y_data.csv', delimiter=',')
data =loaded_data
m = 2
r = 0.2 * np.std(data)
entropy = sample_entropy(data, m, r)
print("Sample Entropy:", entropy)   #Sample Entropy: 0.45665046761628686
