import numpy as np            #logistic代码体现混沌性画图
import matplotlib.pyplot as plt

# Logistic映射参数
r = 3.99  # 控制参数
iterations = 10000  # 迭代次数
x0 = 0.5  # 初始值

# 初始化数组
x = np.zeros(iterations)
x[0] = x0

# 迭代计算
for n in range(1, iterations):
    x[n] = r * x[n-1] * (1 - x[n-1])

# 绘制结果
x_filtered = x[1000:]
plt.figure(figsize=(20, 10))
plt.plot(x_filtered, color='blue')
plt.title('Logistic Map Iteration')
plt.xlabel('Iteration')
plt.ylabel('x')
plt.grid()
plt.show()

#保存
# 保存为 .csv 文件
np.savetxt('logustic_data.csv', x_filtered, delimiter=',')
print("save ok!")

