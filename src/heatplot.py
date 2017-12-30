from numpy import *
import matplotlib.pyplot as plt

def pl(fn):
    with open(fn) as f:
        data = f.readlines()

    data = [float(x) for x in data]

    return data

plt.plot(pl("out/energy.txt"))
plt.plot(pl("out/energy1.txt"))
plt.show()