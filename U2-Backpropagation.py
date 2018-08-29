import numpy as np;

#Input array

X = np.array([][1,0,1,0],[1,0,1,1],[0,1,0,1]])
#output
y = np.array([[1],[1],[0]])

#Sigmoid Funtion
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#deriative of sigmoid Function
def derivatives_sigmoid (x):
    return x * (1-x)

#Variable Initilization
epoch = 5000 #setting training iterations
lr = 0.1
inputlayer_neurons = X.shape[1]
hiddenlayer_neurons = 3
output_neurons = 1

wh = np.random.uniform(size=(inputlayer_neurons,hideenlayer_neurons))
