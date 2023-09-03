import numpy as np

class Neuron:
    def __init__(self, inputs, activation):
            self.weights = np.random.randn(inputs) # Randomly initailizes weights
            self.bias = np.random.randn() # Randomly generates bias
            self.activation = activation

    def forward(self, inputs):
        # Calculate the weighted sum of inputs
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        # Apply the activation function
        return self.activation(weighted_sum)



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Create a neuron with 3 inputs and the sigmoid activation function
neuron = Neuron(inputs=3, activation=sigmoid)

# Input values
inputs = np.array([0.5, 0.2, 0.1])

# Perform forward propagation
# Change to backwards propagation
output = neuron.forward(inputs)

# Display the results
print("\n Neuron:")
print("\t Weights:", neuron.weights)
print("\t Bias:", neuron.bias)
print("\t Input values:", inputs)
print("\t Output:", output)