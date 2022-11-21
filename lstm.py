import numpy as np
import pandas as pd
from math import e, sqrt


def relu(x):
    return x if x > 0 else 0


def sigmoid(x):
    return 1 / (1 + e ** -x)


def sigmoid_diff(x):
    return sigmoid(x) * (1 - sigmoid(x))


def xavier_init(column, row, he=False):
    m = 1
    if he:
        m *= 2
    return np.random.randn(column, row) / sqrt(m/column)


# 참고자료
# https://airsbigdata.tistory.com/195
# https://yngie-c.github.io/deep%20learning/2020/03/17/parameter_init/
class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes,
                 activation_function, learning_rate=0.2, hidden_layers=1, bias=0):
        for n in [input_nodes, hidden_nodes, output_nodes, hidden_layers]:
            if n < 1:
                raise ValueError("All of nodes arguments must be positive integer")
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.hidden_layers = hidden_layers

        if learning_rate > 1 or learning_rate < 0:
            raise ValueError("Learning rate must be between 0 to 1")
        self.lr = learning_rate

        if not callable(activation_function):
            raise TypeError("An activation_function argument is not callable")
        self.activation_function = activation_function

        self.weight_array_ih = xavier_init(self.hnodes, self.inodes)
        self.weight_array_hh = []
        if self.hidden_layers > 1:
            for _ in range(hidden_layers-1):
                self.weight_array_hh.append(xavier_init(hidden_nodes, hidden_nodes))
        self.weight_array_ho = xavier_init(self.onodes, self.hnodes)

    def train(self, inputs_list, targets_list):
        # STEP 0. 행렬 변환
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # STEP 1. 출력 계층의 오차 계산

    def query(self, inputs_list):
        # input to hidden
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.weight_array_ih, inputs)
        hidden_outputs = np.array(list(map(self.activation_function, hidden_inputs)))

        # hidden to hidden
        for i in range(self.hidden_layers - 1):
            hidden_inputs = np.dot(self.weight_array_hh[i], hidden_outputs)
            hidden_outputs = np.array(list(map(self.activation_function, hidden_inputs)))

        # hidden to output
        final_inputs = np.dot(self.weight_array_ho, hidden_outputs)
        final_outputs = np.array(list(map(self.activation_function, final_inputs)))

        return final_outputs


array_sample = np.random.rand(3, 3)

n = NeuralNetwork(2, 3, 1, sigmoid, learning_rate=0.3, hidden_layers=3)

print(n.query([1.0, 0.5]))