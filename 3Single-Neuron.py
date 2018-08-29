from math import*
import numpy as np
from matplotlib import pyplot as plot
def linear(x):
    return x
def threshold(x):
    if x>0:
        return 1
    else:
        return 0
def ramp(x):
    if x>1:
        return 1
    elif x<0:
        return 0
    else:
        return x

def logsig(x):
    return (1/(1+exp(-x)))

x = []
neti = 0
w = []
n = int(input("Enter the size of inputs"))
for i in range(n):
    x.append(float(input("Enter Inputs")))
    w.append(float(input("Enter Weights")))
    neti+=x[i]*w[i]
b = float(input("Enter Value of bias"))
neti = neti + b
ch = int(input("Enter neuron 1. linear 2. threshold 3. Ramp 4. logsigmoid"))
if ch==1:
    out = linear(neti)
elif ch==2:
    out = threshold(neti)
elif ch==3:
    out = ramp(neti)
elif ch==4:
    out = logsig(neti)
else:
    print("Invalid choice of neuron")

print("Output for neuron",out)
