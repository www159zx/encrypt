import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Chen 系统的微分方程
def chen_system(state, t):
    x, y, z = state
    a = 35.0
    b = 3.0
    c = 28.0
    dxdt = a * (y - x)
    dydt = (c-a) * x + c * y - x * z
    dzdt = -b * z + x * y
    return [dxdt, dydt, dzdt]

# 时间参数，设置 dt 为 0.01
dt = 0.01
t = np.arange(0, 20000 + dt, dt)  # 从0到20000，步长为0.01

# 初始状态
initial_state = [0.1, 0.1, 0.1]

# 求解微分方程
solution = odeint(chen_system, initial_state, t)

# 提取解
x = solution[:, 0]
y = solution[:, 1]
z = solution[:, 2]

# 舍弃前1000秒的数据，计算对应的索引
start_index = int(1000 / dt)
x_filtered = x[start_index:]
y_filtered = y[start_index:]
z_filtered = z[start_index:]
t_filtered = t[start_index:]
print("X filtered range:", min(x_filtered), max(x_filtered))
print("Y filtered range:", min(y_filtered), max(y_filtered))
print("Z filtered range:", min(z_filtered), max(z_filtered))

# 绘图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.autoscale(enable=True, axis='both', tight=None)
ax.plot(x_filtered, y_filtered, z_filtered, color='b')
ax.set_title('Chen Chaotic System (After 1000 seconds)')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 设置图形的比例
ax.set_box_aspect([1,1,1])  # 可选，设置比例

plt.show()