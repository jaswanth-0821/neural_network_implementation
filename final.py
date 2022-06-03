

from turtle import forward
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

class Softmax_activation:
    def forward(self,inputs):
        self.exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True)) 
        self.output = self.exp_values/np.sum(self.exp_values,axis = 1,keepdims=True)

class Loss: 
    def calculate(self,otput,y):
        sample_losses = self.forward(otput,y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_Cat_cross(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)

        if len(y_true.shape) ==1:
            correct_conf = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape)==2:
            correct_conf = np.sum(y_pred_clipped*y_true,axis =1)

        neg_log_likelihood = -np.log(correct_conf)
        return neg_log_likelihood


X,y = spiral_data(samples = 1000,classes = 3)

layer1 = Layer_Dense(2,64)
layer2 = Layer_Dense(64,3)
activation1 = Activation_Relu()
softmax_activation1 = Softmax_activation()

step_size = 1e-0
reg = 1e-2
num_examples = X.shape[0]
for i in  range(200):

    layer1.forward(X)
    activation1.forward(layer1.output)
    layer2.forward(activation1.output)
    softmax_activation1.forward(layer2.output)

    loss_function = Loss_Cat_cross()
    loss = loss_function.calculate(softmax_activation1.output,y)
   
    probs1 =activation1.output
    dscores1 = probs1
    dscores1[range(num_examples),y]-=1
    dscores1/=num_examples

    dw1 = np.dot(X.T,dscores1)
    db1 = np.sum(dscores1,axis=0,keepdims=True)

    dw1+= reg*layer1.weights
    
    layer1.weights += -step_size*dw1
    layer1.biases += -step_size*db1
    for j in range(200):
        layer1.forward(X)
        activation1.forward(layer1.output)
        layer2.forward(activation1.output)
        softmax_activation1.forward(layer2.output)

        loss_function = Loss_Cat_cross()
        loss = loss_function.calculate(softmax_activation1.output,y)
     
        probs2 =softmax_activation1.output
        dscores2 = probs2
        dscores2[range(num_examples),y]-=1
        dscores2/=num_examples

        dw2 = np.dot(activation1.output.T,dscores2)
        db2 = np.sum(dscores2,axis=0,keepdims=True)

        dw2+= reg*layer2.weights
        
        layer2.weights += -step_size*dw2
        layer2.biases += -step_size*db2

    if( i==199 and j==199):
        print(f'{loss}')








