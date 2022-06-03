# Activation function

import numpy as np 
import random 
from nnfs.datasets  import spiral_data# it is used to create datasets

X,y = spiral_data(100,3)# creating dataset or input

class Layer_Dense: 
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs,n_neurons) 
        self.biases = np.zeros((1,n_neurons))
    
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

class Activation_Relu:# class that used for activation layer ,here we used relu activation function
    def forward(self,inputs):
        self.output  =np.maximum(0,inputs)# maximum of 0,inputs ,this is the same algorithm used for relu activation

layer1 = Layer_Dense(2,5)
activation1 = Activation_Relu()

layer1.forward(X)
print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)

