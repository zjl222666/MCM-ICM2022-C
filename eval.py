
with open("result.txt", "r") as f:
    action = list(map(lambda x: float(x), f.readlines()))

b2d = action[:1825]
g2d = action[1825:3650]
d2b = action[3650:54]