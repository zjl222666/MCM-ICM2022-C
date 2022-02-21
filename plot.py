
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

fig = plt.figure()
# 创建3d图形的两种方式
# ax = Axes3D(fig)
ax = fig.add_subplot(111, projection='3d')

def change(x):
    for i, c in enumerate(x):
        p = random.random()
        q = random.randint(0,5) /50
        if p > 0.9: x[i] = c * p
        x[i] += q
    return x


X = np.arange(0, 1, 0.02)
Y = np.arange(0, 1, 0.02)
X, Y = np.meshgrid(X, Y)    # x-y 平面的网格
Z = np.sqrt(4 * (1 - X) ** 2 + (1-Y)**(1.2) * 0.5)
# height value
for i, x in enumerate(Z):
    Z[i] = change(x)
Z = np.array(Z) * 190000
# rstride:行之间的跨度  cstride:列之间的跨度
# rcount:设置间隔个数，默认50个，ccount:列的间隔个数  不能与上面两个参数同时出现
#vmax和vmin  颜色的最大值和最小值
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
# zdir : 'z' | 'x' | 'y' 表示把等高线图投射到哪个面
# offset : 表示等高线图投射到指定页面的某个刻度
ax.contourf(X,Y,Z,zdir='z',offset=-2)
# 设置图像z轴的显示范围，x、y轴设置方式相同
ax.set_zlim(-0,400000)
plt.xlabel("cost for bitcoin(%)")
plt.ylabel("cost for gold(%)")
plt.show()
