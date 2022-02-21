import  csv
from pyecharts import options as opts
from pyecharts.charts import Bar, Line, Pie, Boxplot
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType
from sympy import im
from torch import empty
import numpy as np
import random
with open('bitcoin.csv',"r") as f1, open('bitcoin_pre.txt', "r") as f2:
    pre_price = list(map(lambda x: float(x), f2.readline().split()))
    bitcoin_price = list(map(lambda x: float(x[2]), list(csv.reader(f1))[1:]))
pre_price = bitcoin_price[1:201] + pre_price

cur_money = 1000
cur_bit = 0
cost = 0.02
last_money = cur_money
earn = []
num_1 = 0
num_2 = 0
for i, p in enumerate(bitcoin_price):
    if i == len(bitcoin_price) - 1: break
    if cur_money + cur_bit * bitcoin_price[i] - last_money >= 0: num_1 += 1
    else: num_2 += 1
    earn.append(cur_money + cur_bit * bitcoin_price[i] - last_money)
    if bitcoin_price[i+1] < bitcoin_price[i]:
        cur_money += cur_bit * bitcoin_price[i]
        if cur_bit != 0: cur_money *= (1-cost)
        cur_bit = 0
    else:
        if (bitcoin_price[i+1] - bitcoin_price[i]) / bitcoin_price[i] < cost: continue
        cur_bit += cur_money/bitcoin_price[i]
        if cur_money != 0: cur_bit *= (1-cost)
        cur_money = 0

    last_money = cur_money + cur_bit * bitcoin_price[i]
    
arr_var = np.var(earn)
#求标准差
arr_std = np.std(earn,ddof=1)
print("方差为：%f" % arr_var)
print("标准差为:%f" % arr_std)
print(num_1, num_2)
print(cur_money + cur_bit*bitcoin_price[-1]) 
