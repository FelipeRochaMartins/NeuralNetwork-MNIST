import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_layers=[512,512], output_size=10):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.bias = []

        # Setup weights and bias for input layer to first hidden layer
        self.weights.append(0.01 * np.random.randn(self.input_size, self.hidden_layers[0]))
        self.bias.append(np.zeros((1, self.hidden_layers[0])))
        
        # Setup weights and bias for hidden layers
        for i in range(len(self.hidden_layers) - 1):
            self.weights.append(0.01 * np.random.randn(self.hidden_layers[i], self.hidden_layers[i+1]))
            self.bias.append(np.zeros((1, self.hidden_layers[i+1])))

        # Setup weights and bias for last hidden layer to output layer
        self.weights.append(0.01 * np.random.randn(self.hidden_layers[-1], self.output_size))
        self.bias.append(np.zeros((1, self.output_size)))