import pandas as pd
import numpy as np
from tkinter import messagebox
from sklearn.preprocessing import LabelEncoder
import numpy as np
import math
from enum import Enum

# Read the data
dataset = pd.read_excel("Dry_Bean_Dataset.xlsx")


# preprocessing the data
def preprocess(dataset):
    mean_minor_axis_length = dataset['MinorAxisLength'].mean()
    dataset['MinorAxisLength'].fillna(mean_minor_axis_length, inplace=True)
    numeric_columns = dataset.select_dtypes(include=[np.number])  # Select only numerical columns
    # Normalize only the numerical columns
    dataset[numeric_columns.columns] = (numeric_columns - numeric_columns.min()) / (
            numeric_columns.max() - numeric_columns.min())

    # encodeing the label to  0,1,2
    label_encoder = LabelEncoder()
    dataset['Class'] = label_encoder.fit_transform(dataset['Class'])
    # print(dataset)
    return (dataset)


# print(preprocessing(dataset))

def trainTestSplit(dataset):
    # bnakhod awl 50 elly homa kol class
    y1 = dataset[0:50]
    y2 = dataset[50:100]
    y3 = dataset[100:150]

    trainData = pd.concat([y1[0:30], y2[0:30], y3[0:30]]).sample(frac=1, random_state=1).reset_index(drop=True)
    testData = pd.concat([y1[30:50], y2[30:50], y3[30:50]]).sample(frac=1, random_state=1).reset_index(drop=True)

    return trainData, testData


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def tanh(x):
    return (1 - math.exp(-x)) / (1 + math.exp(-x))


class typeOfActivation(Enum):
    tanh = 0,
    sigmoid = 1,


class MLP_model:
    def __init__(self, data: pd.DataFrame, num_layers: int, num_of_neurons: list, eta: float, epoch: int,
                 activation_function: typeOfActivation, bias: bool):
        self.data = data
        self.num_layers = num_layers
        self.num_of_neurons = num_of_neurons

        if activation_function == typeOfActivation.sigmoid:
            self.activation_function = sigmoid
        elif activation_function == typeOfActivation.tanh:
            self.activation_function = tanh
        self.eta = eta
        self.epoch = epoch

        self.train_data, self.test_data = trainTestSplit(dataset)
        self.train_data = self.train_data.sample(frac=1).reset_index(drop=True)
        self.test_data = self.test_data.sample(frac=1).reset_index(drop=True)
        self.x_train = self.train_data[self.train_data.columns[0:-1]]
        self.y_train = self.train_data[self.train_data.columns[5]]

        self.x_test = self.test_data[self.test_data.columns[0:-1]]
        self.y_test = self.test_data[self.test_data.columns[5]]

        # adding the bias to the training and testing datasets
        if bias:
            b = pd.DataFrame(np.ones(len(self.train_data)), columns=['bias'])
            self.x_train = pd.concat([b, self.x_train], axis=1)

            b = pd.DataFrame(np.ones(len(self.test_data)), columns=['bias'])
            self.x_test = pd.concat([b, self.x_test], axis=1)


        # init input weights with random values between -1, 1 for the input layer
        # kda 3mlna matrix of size (num of neurons in the first layer * numb of features in the input )

        # init the weights of the input layer form[-1,1] by  [num of featuers * num of neurons in the first hidden layer]
        w = np.random.uniform(-1, 1, size=(num_of_neurons[0], len(self.x_train.columns)))
        self.weights_arr = [w]

        # weights of the rest hidden layers [-1,1] by [num of neurons of the current layer * num of neurons of the next layer ]
        for i in range(num_layers):
            # case if the first hidden layer hya akher wahda
            if i + 1 >= num_layers:
                break

            d = num_of_neurons[i]
            if bias:
                d += 1

            # init the rest  weights
            # by the num of neurons in the current layer * num of neurons in the nesx layer
            w = np.random.uniform(-1, 1, size=(num_of_neurons[i + 1], d))
            self.weights_arr.append(w)

        # weights of the output layer
        # ehna 3nda 3 output neurons
        # w bnakhod num of neurons in the prev layer
        if bias:
            self.weights_arr.append(np.random.uniform(-1, 1, size=(3, num_of_neurons[num_layers - 1] + 1)))

        else:
            self.weights_arr.append(np.random.uniform(-1, 1, size=(3, num_of_neurons[num_layers - 1])))

        self.bias = bias

    # in forward steps bn3ml 3 hagat :
    # 1- input * weights
    # 2- activation function : res of step 1
    # 3- bn3ml kda for each neuron in each layer fa hn3ml loop on num of layers , num of neurouns

    def feedForward(self, input_row_data):

        k_values = []
        input = [input_row_data]

        for i in range(self.num_layers):

            layer_values = []

            for j in range(self.num_of_neurons[i]):
                # lazem transpose 3shan we're working on matrix fa lzem nzbt deminsions
                net = np.transpose(self.weights_arr[i][j]).dot(input[i])
                value = self.activation_function(net)
                layer_values.append(value)
            k_values.append(layer_values)
            if self.bias:
                layer_values.insert(0, 1)
            # update the input of the next layer by appending the output of the current layer
            input.append(layer_values)

        layer_values = []
        input = k_values[-1]
        for i in range(3):
            # accessing the last layer weight and indixing the neurons
            net = np.transpose(self.weights_arr[-1][i]).dot(input)
            value = self.activation_function(net)
            layer_values.append(value)
        k_values.append(layer_values)

        return k_values

    def feed_Backward(self, k_nets, y):
        output = [0, 0, 0]
        if y == 0:
            output = [1, 0, 0]
        elif y == 1:
            output = [0, 1, 0]
        else:
            output = [0, 0, 1]

        sigma_arr = []
        sigma = []

        for i in range(3):
            sigma.append((output[i] - k_nets[-1][i]) * self.gradient(k_nets[-1][i]))
        sigma_arr.append(sigma)

        for i in reversed(range(self.num_layers)):
            sigma = []
            sigma_arr
            for j in range(self.num_of_neurons[i]):
                s = 0
                for k in range(len([0])):
                    s += self.weights_arr[i + 1][k][j] * sigma_arr[0][k]
                sigma.append(s * self.gradient(k_nets[i][j]))
            sigma_arr.insert(0, sigma)
        return sigma_arr

    def weightsUpdate(self, sigmas, k_nets, inputRowData):
        # delta W = eta * sigmoid * output
        k_nets.insert(0, inputRowData)
        # loop over the layers  , then on layer neurons , then on each neuron weights
        for i in range(len(self.weights_arr)):
            for j in range(len(self.weights_arr[i])):
                for k in range(len(self.weights_arr[i][j])):
                    self.weights_arr[i][j][k] += self.eta * sigmas[i][j] * k_nets[i][k]

    def gradient(self, x):
        # calc drivatives of the activations
        if self.activation_function == sigmoid:
            return x * (1 - x)
        else:
            return 1 - (x ** 2)

    def Train(self):
        accuracy = 0
        for i in range(self.epoch):
            for j in range(len(self.x_train)):
                k_nets = self.feedForward(self.x_train.values[j])
                y = k_nets[-1]

                sigmas = self.feed_Backward(k_nets, self.y_train.values[j])
                self.weightsUpdate(sigmas, k_nets, self.x_train.values[j])

                y_indx = -1
                if y[0] > y[1] and y[0] > y[2]:
                    y_indx = 0
                elif y[1] > y[0] and y[1] > y[2]:
                    y_indx = 1
                elif y[2] > y[1] and y[2] > y[0]:
                    y_indx = 2

                if self.y_train.values[j] == 0 and y_indx == 0:
                    accuracy += 1
                elif self.y_train.values[j] == 1 and y_indx == 1:
                    accuracy += 1
                elif self.y_train.values[j] == 2 and y_indx == 2:
                    accuracy += 1

        accuracy /= len(self.x_train)

    def Test(self):
        accuracy = 0
        confusion_matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        y_index = -1
        t_index = -1
        for i in range(len(self.y_test)):
            f_nets = self.feedForward(self.x_test.values[i])
            y = f_nets[-1]
            t = [0, 0, 0]

            if y[0] > y[1] and y[0] > y[2]:
                y = [1, 0, 0]
                y_index = 0
            elif y[1] > y[0] and y[1] > y[2]:
                y = [0, 1, 0]
                y_index = 1
            elif y[2] > y[1] and y[2] > y[0]:
                y = [0, 0, 1]
                y_index = 2

            if self.y_test.values[i] == 0:
                t = [1, 0, 0]
                t_index = 0
            elif self.y_test.values[i] == 1:
                t = [0, 1, 0]
                t_index = 1
            elif self.y_test.values[i] == 2:
                t = [0, 0, 1]
                t_index = 2

            if t == y:
                accuracy += 1

            confusion_matrix[y_index][t_index] += 1

        # calculate testing accuracy
        accuracy = (accuracy / len(self.y_test)) * 100

        print(f'Testing Accuracy: {accuracy:.2f}%')
        print(f'Confusion Matrix:')
        print(f'\t\tone\t|\ttwo\t|\tthree')
        print(f'one:\t{confusion_matrix[0][0]}\t|\t{confusion_matrix[0][1]}\t|\t{confusion_matrix[0][2]}\t')
        print(f'two:\t{confusion_matrix[1][0]}\t|\t{confusion_matrix[1][1]}\t|\t{confusion_matrix[1][2]}\t')
        print(f'three:\t{confusion_matrix[2][0]}\t|\t{confusion_matrix[2][1]}\t|\t{confusion_matrix[2][2]}\t')
        print('--------------------------------')