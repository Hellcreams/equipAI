import numpy as np
import pandas as pd
from math import e, sqrt


def relu(x):
    return x if x > 0 else 0


def sigmoid(x):
    return 1 / (1 + e ** -x)


def xavier_init(column, row, he=False):
    m = 1
    if he:
        m *= 2
    return np.random.randn(column, row) / sqrt(m / column)


class Perceptron:
    def __init__(self, activation_function, diff_increment=10**-8):
        self.x = None
        self.h = diff_increment
        self.function = np.vectorize(activation_function)

    def forward(self, x):
        self.x = np.array(x, dtype="float64")
        return self.function(*x)

    def backward(self, dout):
        xh = np.array(self.x, dtype="float64")
        dx = []
        for i in range(len(xh)):
            xh[i] += self.h
            d = (self.function(*xh) - self.function(*self.x)) / self.h
            dx.append(d)
            xh[i] -= self.h

        return np.array(dx) * dout


class Gate:
    def __init__(self, units, function, bias, diff_increment=10**-8):
        self.perceptron = Perceptron(function, diff_increment)
        self.input_nodes = units
        self.x_weight = xavier_init(units, 1)
        self.h_weight = np.zeros((units, 1))
        self.bias = bias

    def forward(self, x_t, h_t):
        return self.perceptron.forward([self.x_weight, x_t, self.h_weight, h_t, self.bias])

    def backward(self, dout):
        return self.backward(dout)

    def update_weight(self, x_weight, h_weight):
        self.x_weight += x_weight
        self.h_weight += h_weight


class LSTM:
    def __init__(self):



# 참고자료
# https://airsbigdata.tistory.com/195
# https://yngie-c.github.io/deep%20learning/2020/03/17/parameter_init/
# https://webnautes.tistory.com/1655
# https://docs.likejazz.com/lstm/
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

        self.bias = bias

        self.weight_array_ih = xavier_init(self.hnodes, self.inodes)
        self.weight_array_hh = []
        if self.hidden_layers > 1:
            for _ in range(hidden_layers - 1):
                self.weight_array_hh.append(xavier_init(hidden_nodes, hidden_nodes))
        self.weight_array_ho = xavier_init(self.onodes, self.hnodes)

    def query(self, inputs_list):
        # input to hidden
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.weight_array_ih, inputs)
        hidden_outputs = np.array(list(map(self.activation_function, hidden_inputs))) + self.bias

        # hidden to hidden
        for i in range(self.hidden_layers - 1):
            hidden_inputs = np.dot(self.weight_array_hh[i], hidden_outputs)
            hidden_outputs = np.array(list(map(self.activation_function, hidden_inputs))) + self.bias

        # hidden to output
        final_inputs = np.dot(self.weight_array_ho, hidden_outputs)
        final_outputs = np.array(list(map(self.activation_function, final_inputs)))

        return final_outputs

    def train(self, inputs_list, targets_list):
        # STEP 0. 행렬 변환
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # query와 동일
        hidden_inputs = [np.dot(self.weight_array_ih, inputs)]
        hidden_outputs = [np.array(list(map(self.activation_function, hidden_inputs))) + self.bias]
        type(hidden_outputs)
        for i in range(self.hidden_layers - 1):
            hidden_inputs.append(np.dot(self.weight_array_hh[i], hidden_outputs))
            hidden_outputs.append(np.array(list(map(self.activation_function, hidden_inputs[-1]))) + self.bias)
        final_inputs = np.dot(self.weight_array_ho, hidden_outputs)
        final_outputs = np.array(list(map(self.activation_function, final_inputs)))

        # STEP 1. 출력 계층의 오차 계산
        output_errors = targets - final_outputs

        # STEP 2. 은닉 계층의 역전파된 오차 계산
        hidden_errors = [np.dot(self.weight_array_ho.T, output_errors)]
        for i in range(1, self.hidden_layers):
            hidden_errors.append((np.dot(self.weight_array_hh[-i].T, hidden_errors[-1])))

        # STEP 3. 가중치 업데이트
        self.weight_array_ho = self.lr * np.dot((output_errors * sigmoid_diff(final_outputs)), hidden_outputs[-1].T)
        for i in range(1, self.hidden_layers):
            self.weight_array_hh[-i] = self.lr * np.dot((hidden_errors[-i] * sigmoid_diff(hidden_outputs[-i])),
                                                        hidden_outputs[-(i + 1)].T)
        self.weight_array_ih = self.lr * np.dot((hidden_errors[0] * sigmoid_diff(hidden_outputs[0])), inputs.T)


class LSTM:
    def __init__(self):
        pass




a = Perceptron(lambda x, y: x + y)
print(a.forward([2, 10]))
l = a.backward(2)
print(l[0], l[1])
