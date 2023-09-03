
import numpy as np

# Define a neuron class
class Neuron:

    # Initalizes weights and biases
    def __init__ (self, input_size):
        self.weights = np.random.uniform(-1, 1, (input_size,))
        self.bias = 0

    # Activation function
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    # Forward propagation
    def forward(self, x):
        # Calculate the weighted sum and apply the sigmoid activation function
        z = np.dot(x, self.weights) + self.bias
        return self.sigmoid(z)
    
class NeuralNetwork:
    def __init__ (self, input_size, hidden_layer_size, output_size):
        self.hidden_layer = [Neuron(input_size) for _ in range(hidden_layer_size)]
        self.output_layer = [Neuron(hidden_layer_size) for _ in range(output_size)]

