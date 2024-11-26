import numpy as np
import matplotlib.pyplot as plt

def henon_map(x, y, a=1.4, b=0.3):
    """Henon map function."""
    return 1 - a * x**2 + y, b * x

# 设置初始条件和迭代次数
x0, y0 = 0.1, 0.3
iterations = 10000

# 用于存储结果
x_values = []
y_values = []

# 迭代Henon映射
x, y = x0, y0
for _ in range(iterations):
    x, y = henon_map(x, y)
    x_values.append(x)
    y_values.append(y)

# 绘制Henon吸引子
#x_filtered = x[1000:]
x_values_new= x_values[1000:]
y_values_new= y_values[1000:]
plt.figure(figsize=(15, 10))
plt.plot(x_values_new, y_values_new, 'b.', markersize=1)
plt.title('Henon Map')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()

np.savetxt('henon_x_data.csv', x_values_new, delimiter=',')
print("save ok!")
np.savetxt('henon_y_data.csv', y_values_new, delimiter=',')
print("save ok!")
