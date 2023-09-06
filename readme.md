# Neural Netowrks

This project is a introdution to neural networks and how they operate. This project is a multi-layer perceptron aimed at recognising hand drawn digits. I used 3Blue1Browns neural network playlist as a reference point for this project. The data set used is the MNIST database of hand drawn digits (http://yann.lecun.com/exdb/mnist/). Hopefully be able to tie this in with hardware in the future to create some cool projects.

## Creating the Network

A neuron holds all the weights connecting it to the next layer, in an array. It also holds a bais parameter as a float. The network stores neurons in different arrays, referred to as layers. The number of weights a neuron holds is determined by the number of neurons in the next layer. This prevents the need for pointers, as the index of the next layer aligns with the weights. The activation number isn't stored in the neuron but calculated during the forward propagation and stored in an array. The input matrix is treated as the connections from the first layer.


## Training

Gradient descent
back prop

## Optimization

stohastic back prop

different acivation functions:
    - Sigmoid
    - Hyperbolic Tangent
    - ReLU
    - Leaky ReLU
    - PReLU
    - ELU


## Visualization
Graphviz
