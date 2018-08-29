from math import*
import numpy as np
from matplotlib import pyplot as plt

def threshold(x):
    if x>0:
        return 1
    else:
        return 0

x = []
neti = 0
w = []
n = int(input("Enter size of inputs"))
for i in range(n):
    x.append(float(input("Enter Inputs")))
    w.append(float(input("Enter Weights")))
    neti+=x[i]*w[i]
b = float(input("Enter Value of bias"))
neti = neti + b
out = threshold(neti)

print("Out put for Maculloch pit neuron",out)
