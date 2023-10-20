import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
np.random.seed(1)

def synaptic_weights(n_inputs):
    """ Synaptic weights:
    - Random value for weight inicialisation.
    """
    w=2*np.random.random((n_inputs,1))-1
    return w

def sigmoid(x):
    """ Sigmoid Function:
    - Activation function for the neuron.
    """
    # Sigmoid
    phi=1/(1+np.exp(-x))
    return phi

def sigmoid_prime(x):
    """ Derivative of the Sigmoid Function:
    - Used for the error adjustment as the 
    gradient desccent function. To minimise
    the error 
    """
    phi_prime=x*(1-x)
    return phi_prime

def error_weighted(y,y_hat):
    """ Error:
    - Get the error by substracting the 
    actual output from the 
    """
    error=y-y_hat
    return error
def adjustment(error,output):
    weight_adjustment=error*sigmoid_prime(output)
    return weight_adjustment



training_inputs=np.array([[0,1,0,0],
                          [1,0,1,1],
                          [1,1,0,0],
                          [0,1,1,0]])
training_outputs=np.array([[0,1,1,0]]).T
print('Training Inputs')
print(training_inputs)
print('Training Outputs')
print(training_outputs)

error_list=[]
iteration_list=[]
weights=synaptic_weights(4)
for iteration in range(200000):
    iteration_list.append(iteration)
    input_layer=training_inputs
    y_hat=sigmoid(np.dot(input_layer,weights))
    error=error_weighted(training_outputs,y_hat)
    error_list.append(math.log(sum(error)**2))
    weight_adjustment=adjustment(error,y_hat)
    weights+=np.dot(input_layer.T,weight_adjustment)

print('Y_HAT',y_hat)





x_p=np.linspace(-6,6,100)
plt.figure()
plt.grid()
plt.plot(x_p,sigmoid(x_p))
plt.scatter([0,0,0,0],y_hat)
plt.figure()
plt.plot(x_p,sigmoid_prime(x_p))
plt.figure()
plt.scatter(iteration_list,error_list)
plt.show()