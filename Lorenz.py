import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd

# 洛伦兹系统的方程
def lorenz_system(state, t, sigma, beta, rho):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# 参数
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

# 初始条件
initial_state = [1.0, 1.0, 1.0]

# 时间点
t = np.linspace(0, 50, 10000)  # 从0到50秒，共10000个点

# 求解微分方程
solution = odeint(lorenz_system, initial_state, t, args=(sigma, beta, rho))

# 获取解的x, y, z
x, y, z = solution.T
save_x=x[1000:]
save_y=y[1000:]
save_z=z[1000:]

# 绘图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(save_x, save_y, save_z, lw=0.5, color='b')

# 设置标签
ax.set_title('Lorenz Attractor', fontsize=20)
ax.set_xlabel('X axis', fontsize=15)
ax.set_ylabel('Y axis', fontsize=15)
ax.set_zlabel('Z axis', fontsize=15)

plt.show()

df = pd.DataFrame({'x': save_x, 'y': save_y, 'z': save_z})

# 保存为 CSV 文件
df.to_csv('lorenz_data.csv', index=False)
