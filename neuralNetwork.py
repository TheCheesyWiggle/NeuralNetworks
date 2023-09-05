
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
        # Populating the layers with neurons and establishing connections between layers for forwards and backwards propagation
        self.hidden_layer = [Neuron(input_size) for _ in range(hidden_layer_size)]
        self.output_layer = [Neuron(hidden_layer_size) for _ in range(output_size)]

    # Forward propagation
    def forward(self, x):
        # Calculates the actvation of each neuron in the hidden layer using the input list
        hidden_outputs =[neuron.forward(x) for neuron in self.hidden_layer]
        # Calculates the activation of each neuron in the output layer using the previous hidden layers' neuron list
        output  = [neuron.forward(hidden_outputs) for neuron in self.output_layer]
        return output
    
    # Backward propagation
    # y is the desired output 
    # x is the input data (Matrix prepresentation of handwritten digit)
    # learning rate controls the step size
    def back_prop(self, x, y, learning_rate):
        # Do a forward pass to get an output array
        output = self.forward(x)
        # Calucate the difference from you desired output (loss)
        loss = np.array(output) - y
        # Propagate back through the network for each layer with respect to the current neuron
        # Output layer
        for i, neuron in enumerate(self.output_layer):
            # C0 sum of a(L)j - yj squared

