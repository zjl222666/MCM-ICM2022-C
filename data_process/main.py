import csv


with open('bitcoin.csv',"r") as f, open('bitcoin.csv',"r") as f1, open('bitcoin_pre.txt', "r") as f2, open('gold_pre.txt') as f3:
    data = list(map(lambda x:x[1] if len(x[1]) <= 8 else x[1][2:] if x[1][2] == '1' else x[1][3:], list(csv.reader(f1))[1:]))

    bit_pre = list(f2.readline().split())
    gold_pre = list(f3.readline().split())

with open('trans.csv', 'w', newline = '') as f:
    writer = csv.writer(f)
    for a in data:
        writer.writerow([str(a)])

