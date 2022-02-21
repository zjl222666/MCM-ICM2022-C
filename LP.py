'''
this model used to solve the LP problem when all the data is permitted to help models make action
'''
import copy
import csv
import numpy as np
from scipy.optimize import linprog
leng = 1825

with open('bitcoin.csv',"r") as f1, open('GOLD.csv', "r") as f2:
    gold_price = list(map(lambda x: float(x[2]), list(csv.reader(f2))[1:]))
    bitcoin_price = list(map(lambda x: float(x[2]), list(csv.reader(f1))[1:]))

gold_cost = 0.01
bitcoin_cost = 0.02
gold_price = gold_price[:leng]
bitcoin_price = bitcoin_price[:leng]
S_dollar = [{}]
S_gold = [{}]
S_bitcoin= [{}]

S_dollar[0]['b2d'] = [0] * leng
S_dollar[0]['g2d'] = [0] * leng
S_dollar[0]['d2b'] = [0] * leng
S_dollar[0]['d2g'] = [0] * leng
S_gold[0]['g2d'] = [0] * leng
S_gold[0]['d2g'] = [0] * leng
S_bitcoin[0]['b2d'] = [0] * leng
S_bitcoin[0]['d2b'] = [0] * leng

def S():
    i = int(len(S_dollar)) - 1
    S_dollar.append(copy.deepcopy(S_dollar[-1]))
    S_gold.append(copy.deepcopy(S_gold[-1].copy()))
    S_bitcoin.append(copy.deepcopy(S_bitcoin[-1].copy()))
    S_dollar[-1]['b2d'][i] += bitcoin_price[i] * (1-bitcoin_cost)
    S_dollar[-1]['g2d'][i] += gold_price[i] * (1-gold_cost)
    S_dollar[-1]['d2b'][i] -= 1
    S_dollar[-1]['d2g'][i] -= 1
    S_gold[-1]['d2g'][i] += (1-gold_cost) / gold_price[i]
    S_gold[-1]['g2d'][i] -= 1
    S_bitcoin[-1]['d2b'][i] += (1-bitcoin_cost) / bitcoin_price[i]
    S_bitcoin[-1]['b2d'][i] -= 1

for _ in range(leng-1):
    S()



c = list(map(lambda x: x[0]+x[1]*bitcoin_price[-1], zip(S_dollar[-1]['b2d'], S_bitcoin[-1]['b2d']))) + \
    list(map(lambda x: x[0]+x[1]*gold_price[-1], zip(S_dollar[-1]['g2d'], S_gold[-1]['g2d']))) + \
    list(map(lambda x: x[0]+x[1]*bitcoin_price[-1], zip(S_dollar[-1]['d2b'], S_bitcoin[-1]['d2b']))) + \
    list(map(lambda x: x[0]+x[1]*gold_price[-1], zip(S_dollar[-1]['d2g'], S_gold[-1]['d2g'])))

A = []
B = []
index = -1
for d, g, b in zip(S_dollar[:-1], S_gold[:-1], S_bitcoin[:-1]):
    index += 1
    cur_dollars = d['b2d'] + d['g2d'] + d['d2b'] + d['d2g']
    cur_dollars[index + 2*leng] -= 1
    cur_dollars[index + 3*leng] -= 1
    cur_gold =  [0] * leng + g['g2d'] +  [0] * leng + g['d2g']
    cur_gold[index + leng] -= 1
    cur_bit = b['b2d'] +  [0] * leng + b['d2b'] +  [0] * leng
    cur_bit[index] -= 1
    A.append(-np.array(cur_dollars))
    A.append(-np.array(cur_gold))
    A.append(-np.array(cur_bit))
    B += [1000, 0, 0]

for i in range(4*leng):
    B.append(0)
    tmp = np.zeros(leng * 4)
    tmp[i] = -1
    A.append(tmp)


A = np.array(A)
B = np.array(B)
c = -np.array(c)
output_file  = open("result.txt", 'w')
print('begin LP.... ')
res = linprog(c, A_ub=A, b_ub = B)
print('Optimal value: ', round(-res.fun, ndigits=4), file=output_file)
for x in res.x:
    print(x, file=output_file)
print(res.nit, res.message)



        
