import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from decimal import Decimal, getcontext
import math

# Chen 系统的微分方程
def chen_system(state, t):
    x, y, z, w= state
    a = 35.0
    b = 3.0
    c = 12.0
    d=7.0
    r=0.58     #0.085~0.798
    dxdt = a * (y - x)+w
    dydt = (d-z) * x + c * y
    dzdt = -b * z + x * y
    dwdt = y * z + r * w
    return [dxdt, dydt, dzdt, dwdt]

# 时间参数，设置 dt 为 0.01
dt = 0.01
t = np.arange(0, 10000 + dt, dt)  # 从0到20000，步长为0.01

# 初始状态
initial_state = [1, 1.1, 1.2, 1.3]

# 求解微分方程
solution = odeint(chen_system, initial_state, t)

# 提取解
x = solution[:, 0]
y = solution[:, 1]
z = solution[:, 2]
w = solution[:, 3]

# 舍弃前1000秒的数据，计算对应的索引
start_index = int(1500 / dt)
x_filtered = x[start_index:]
y_filtered = y[start_index:]
z_filtered = z[start_index:]
w_filtered = w[start_index:]
t_filtered = t[start_index:]

mod_array_x = [math.floor(x_val * 10**16) % 8 for x_val in x_filtered]
mod_array_y = [math.floor(y_val * 10**16) % 8 for y_val in y_filtered]
mod_array_z = [math.floor(z_val * 10**16) % 8 for z_val in z_filtered]
mod_array_w = [math.floor(w_val * 10**16) % 8 for w_val in w_filtered]
print(mod_array_x)
# 绘图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
#ax.plot(x_filtered, y_filtered, color='b')
ax.plot(x_filtered, w_filtered, color='b',marker='.', linestyle='')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')

plt.show()


print("save ok!")

# 输出结果
#print(decimal_13th_digits)