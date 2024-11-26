import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from decimal import Decimal, getcontext

# Chen 系统的微分方程
def chen_system(state, t):
    x, y, z, w= state
    a = 35.0
    b = 3.0
    c = 12.0
    d=7.0
    r=0.7     #0.085~0.798
    dxdt = a * (y - x)+w
    dydt = (d-z) * x + c * y
    dzdt = -b * z + x * y
    dwdt = y * z + r * w
    return [dxdt, dydt, dzdt, dwdt]

# 时间参数，设置 dt 为 0.01
dt = 0.01
t = np.arange(0, 20000 + dt, dt)  # 从0到20000，步长为0.01

# 初始状态
initial_state = [0.1, 0.1, 0.1, 0.1]

# 求解微分方程
solution = odeint(chen_system, initial_state, t)

# 提取解
x = solution[:, 0]
y = solution[:, 1]
z = solution[:, 2]
w = solution[:, 3]

# 舍弃前1000秒的数据，计算对应的索引
start_index = int(1000 / dt)
x_filtered = x[start_index:]
y_filtered = y[start_index:]
z_filtered = z[start_index:]
w_filtered = w[start_index:]
t_filtered = t[start_index:]
print("X filtered range:", min(x_filtered), max(x_filtered))
print("Y filtered range:", min(y_filtered), max(y_filtered))
print("Z filtered range:", min(z_filtered), max(z_filtered))

# 绘图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
#ax.plot(x_filtered, y_filtered, color='b')
ax.plot(x_filtered, w_filtered, color='b',marker='.', linestyle='')
ax.set_title('Chen Chaotic System (After 1000 seconds) on XY plane')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')

plt.show()

# 设置浮动点的精度，保留15位有效数字
getcontext().prec = 15

# 示例 NumPy 数组
result = []

# 使用 zip 函数将四个数组按位置组合，然后扩展到结果数组中
for a, b, c, d in zip(x_filtered, y_filtered, z_filtered, w_filtered):
    result.extend([a, b, c, d])

nums = result

# 定义一个函数来获取小数部分的第13位数字
def get_13th_decimal_digit(num):
    # 转换为Decimal类型
    num = Decimal(str(num))

    # 获取小数部分
    decimal_part = num - int(num)  # 取得小数部分

    # 将小数部分乘以10的13次方
    scaled = decimal_part * 10 ** 13

    # 返回小数部分的第13位数字
    return int(scaled) % 10  # 取最后一位

# 处理整个数组，提取每个元素的小数部分的第13位数字
decimal_13th_digits = [get_13th_decimal_digit(num) for num in nums]

#np.savetxt('chen_max.csv', decimal_13th_digits, delimiter=',')



np.savetxt('output.csv', x_filtered, delimiter=',')






print("save ok!")

# 输出结果
#print(decimal_13th_digits)