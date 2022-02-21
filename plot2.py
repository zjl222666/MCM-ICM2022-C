import random
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
# 保证图片在浏览器内正常显示
x = []
y = []
for i in range(10):
    for j in range(10):
        t = random.random()
        t2 = random.random()
        x.append((i+t)/10)
        y.append((j+t2)/10)
for i in range(10):
    for j in range(10):
        t = random.randint(max(0, 2-i)*100, min(10, 2+i)*100)/1000
        t2 = random.randint(max(0, 1-i)*100, min(10, 1+i)*100)/1000
        x.append(t)
        y.append(t2)
        
x = np.array(x)
y = np.array(y)
plt.xlabel("cost for bitcoin(%)")
plt.ylabel("cost for gold(%)")
plt.scatter(x, y)
plt.show()