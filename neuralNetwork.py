
import numpy as np

# Define a neuron class
class Neuron:
    # Initalizes weights and biases
    def __init__ (self, input_size):
        self.weights = np.random.uniform(-1, 1, (input_size))
        self.bias = 0

    # Activation function
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    # Activation Derivative
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    # Forward propagation
    def forward(self, x):
        # Calculate the weighted sum and apply the sigmoid activation function
        z = np.dot(x, self.weights) + self.bias
        print(z)
        return self.sigmoid(z)
    
class NeuralNetwork:
    def __init__ (self, input_size, hidden_layer_size, output_size):
        # Populating the layers with neurons and establishing connections between layers for forwards and backwards propagation
        self.hidden_layer = [Neuron(input_size) for _ in range(hidden_layer_size)]
        self.output_layer = [Neuron(hidden_layer_size) for _ in range(output_size)]
        # array of each cost to calc average cost (opitmize later)
        self.costs = []

    # Forward propagation
    # x is the input matrix
    def forward(self, x):
        # Calculates the actvation of each neuron in the hidden layer using the input list
        hidden_outputs =[neuron.forward(x) for neuron in self.hidden_layer]
        # Calculates the activation of each neuron in the output layer using the previous hidden layers' neuron list
        output  = [neuron.forward(hidden_outputs) for neuron in self.output_layer]
        return output
    
def backward(self, x, y, learning_rate):
        # Calculate the forward pass
        hidden_outputs = [neuron.forward(x) for neuron in self.hidden_layer]
        outputs = [neuron.forward(hidden_outputs) for neuron in self.output_layer]

        # Calculate the error in the output layer
        output_errors = [y_i - output_i for y_i, output_i in zip(y, outputs)]

        # Update the weights and biases in the output layer
        for i, neuron in enumerate(self.output_layer):
            for j, weight in enumerate(neuron.weights):
                print("hi")
                #update weights