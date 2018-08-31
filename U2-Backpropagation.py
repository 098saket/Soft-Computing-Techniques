import numpy as np;
import math

#Input array

X = np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
#output
y = np.array([[1],[1],[0]])

#Sigmoid Funtion
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#deriative of sigmoid Function
def derivatives_sigmoid (x):
    return x * (1-x)

#Variable Initilization
Epoch = 5000 #setting training iterations
lr = 0.1
inputlayer_neurons = X.shape[1]
hiddenlayer_neurons = 3
output_neurons = 1

wh = np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh = np.random.uniform(size=(1,hiddenlayer_neurons))
wout  =np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout = np.random.uniform(size=(1,output_neurons))

for i in range(Epoch):
    print('Epoch',i)
    hidden_layer_input1 = np.dot(X,wh)
    hidden_layer_input = sigmoid(hidden_layer_input)
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1 = np.dot(hiddenlayer_activations.wout)
    output_layer_input = output_layer_input1 + bout
    output = sigmoid(output_layer_input)

    E = y-output
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
    d_output = E * slope_output_layer

    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    wout = wout + hiddenlayer_activations.T.dot(d_output) * Ir
    bout = bout + np.sum(d_output,axis=0,keepdims = True) * Ir
    wh = wh + X.T.dot(d_hiddenlayer) * Ir
    bh = bh + np.sum(d_hiddenlayer,axis=0,keepdims=True) * Ir
print("Target Values")
print(y)
print('output_values')
print(output)
print('E = Target - output')
print(E)
