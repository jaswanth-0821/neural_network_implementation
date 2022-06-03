#softmax activation function
import numpy as np
import math 
import random 
from nnfs.datasets  import spiral_data


E = math.e

class Layer_Dense: 
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs,n_neurons) 
        self.biases = np.zeros((1,n_neurons))
    
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

class Activation_Relu:
    def forward(self,inputs):
        self.output  =np.maximum(0,inputs)

class Softmax_activation:#creating Softmax activation algorithm 
    def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True)) 
        self.output = exp_values/np.sum(exp_values,axis = 1,keepdims=True)


X,y = spiral_data(samples = 100,classes =64)
# creating all layers ,relu activation ,softmax activation functions
layer1 = Layer_Dense(2,64)
layer2 = Layer_Dense(64,3)
activation1 = Activation_Relu()
softmax_activation1 = Softmax_activation()
# keeping the spiral data in neural network
layer1.forward(X)
activation1.forward(layer1.output)# The output of layer1 is sent to relu activation function 
layer2.forward(activation1.output)# the output of relu activation is connected second neural network 
softmax_activation1.forward(layer2.output)# the output of second neural network is connected to softmax activation layer

print(activation1.output.shape)
print(layer1.output.shape)





