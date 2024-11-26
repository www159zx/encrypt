import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from decimal import Decimal, getcontext
import math

df = pd.read_csv('logistic_test.csv', header=None)
# 将DataFrame转换为NumPy数组
data = df.values.flatten()

image = Image.open('pic.jpg').convert('L')
image_matrix = np.array(image)

m=len(image_matrix)
n=len(image_matrix[0])
required_size = m * n
len(data) >= required_size
sliced_array = large_array[:required_size]  # 截取前 required_size 个元素
matrix = sliced_array.reshape(m, n)  # 这是A3矩阵

DNA_ENCODING = {
    '00': 'A',
    '01': 'T',
    '10': 'C',
    '11': 'G'
}