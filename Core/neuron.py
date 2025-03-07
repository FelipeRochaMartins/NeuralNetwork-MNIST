import pandas as pd
import numpy as np

class Neuron:
    def __init__(self, weights, bias, inputs):
        self.weights = weights
        self.bias = bias
        self.inputs = inputs
        self.output = 0

    def forward(self):
        self.output = np.dot(self.inputs, self.weights) + self.bias
        return self.output