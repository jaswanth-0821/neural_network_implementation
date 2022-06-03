#batches ,Layers and objects 
import numpy as np # importing numpy as np
import random 

input = [[1,2,3,2.5],
         [2.0,5.0,-1.0,2.0],
         [-1.5,2.7,3.3,-0.8]]

class Layer_Dense: # Creating a object named Layer_Dense so we can use this any number of times.
    def __init__(self,n_inputs,n_neurons):# inititing init function with inputs as n_inputs,n_neurons which are used for the shape of weights
        self.weights = 0.10*np.random.randn(n_inputs,n_neurons) # defining weights with shape as n_inputs,n_neurons
        self.biases = np.zeros((1,n_neurons))# defining biases with shape [1,n_neurons] and initial values as zeros
    
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

layer1 = Layer_Dense(4,64)# create a layer with shape as[4,5] that means it as 4 inputs and 5 neurons 
layer2 = Layer_Dense(64,2)# create a layer with shape [5,2]
# we need to take care that the first one present layer and seoncd one of previous layer should be equal
layer1.forward(input)
layer2.forward(layer1.output)

#print(layer1.output.shape)
print(layer2.output)
