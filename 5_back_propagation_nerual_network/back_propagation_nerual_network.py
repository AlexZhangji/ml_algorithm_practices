"""
In the directory gestures, there is a set of images1 that display "down" gestures (i.e., thumbs-down images) or
other gestures. In this assignment, you are required to implement the Back Propagation algorithm for Feed Forward
Neural Networks to learn down gestures from training images available in downgesture_train.list. The label of an
image is 1 if the word "down" is in its file name; otherwise the label is 0. The pixels of an image use the gray
scale ranging from 0 to 1. In your network, use one input layer, one hidden layer of size 100, and one output node.
Use the value 0.1 for the learning rate. For each perceptron, use the sigmoid function (s) = 1/(1+eƟ-s). Use 1000
training epochs; initialize all w randomly between -1000 to 1000 (you can also choose your own initialization approach,
as long as it works); and then use the trained network to predict the labels for the gestures in the test images
available in downgesture_test.list. For the error function, use the standard least square error function.
Output your predictions and accuracy.
"""

import time
import cv2
import numpy as np


def get_training_set(fname='downgesture_train.list'):
    training_data = []
    labels = []

    with open(fname) as f:
        training_files = [line.rstrip() for line in f if line]

    for file_dir in training_files:
        max_gray_val = 255.0  # default max gray val to 255. can also read in every time.
        pgm_data = cv2.imread(file_dir, -1)  # read pgm with OpenCV, 30 * 32 ndarray
        # convert to 1-d array and normalize it with max gray val
        normalized_pgm = np.array(pgm_data).flatten() / max_gray_val
        label = 0
        if 'down' in file_dir:
            label = 1

        training_data.append(normalized_pgm)
        labels.append(label)

    return np.array(training_data), np.array(labels), training_files


class FeedForwardNeuralNetwork(object):
    def __init__(self, input_size, n_input_layers=1, n_hidden_layer=1, hidden_layer_size=100, n_output_node=1,
                 learning_rate=0.1, n_epochs=1000):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.n_output_node = n_output_node
        self.n_hidden_layer = n_hidden_layer
        self.n_input_layers = n_input_layers
        self.n_epochs = n_epochs  # num of times to run and update whole weights
        self.learning_rate = learning_rate

        self.first_layer_res = None
        self.hidden_layer_res = None
        # init weights within [-1000, 1000] and two layers
        self.weights = np.random.uniform(-1, 1, size=(self.input_size + 1, self.hidden_layer_size))  # 184*100
        self.hidden_weights = np.random.uniform(-1, 1, size=(self.hidden_layer_size, self.n_output_node))  # 100*1

        # # init bias
        # self.bias = np.random.uniform(-1000, 1000, size=1)

    def train_network(self, X, y):
        # add bias to X.
        bias = np.ones((184, 1))
        X = np.concatenate((X, bias), axis=1)

        for _ in range(self.n_epochs):

            # training using SGD, with mini_batch_size customized
            batch_x, batch_y = self.SGD(X, y, mini_batch_size=17)
            for cur_x, cur_y in zip(batch_x, batch_y):
                # get the final prediction for the input with current weights
                prediction = self.feed_forward(cur_x)

                # updates weights by the error generated for each level.
                self.back_propagate(prediction, cur_x, cur_y)

    def SGD(self, X, y, mini_batch_size):
        """
            return small portion of paired X and y instead of going through all of them
        """
        # randomly choosing mini_batch_size number of X and y elements
        batch_index_list = list(np.random.random_integers(0, len(X) - 1, size=mini_batch_size))
        return [X[i] for i in batch_index_list], [y[i] for i in batch_index_list]

    def feed_forward(self, x):
        # first level results, shape 100*1
        self.first_layer_res = self.sigmoid(np.dot(x, self.weights))
        # hidden layer results, shape 1*1
        self.hidden_layer_res = self.sigmoid(np.dot(self.first_layer_res, self.hidden_weights))
        return self.hidden_layer_res

    def back_propagate(self, prediction, cur_x, cur_y):
        error = cur_y - prediction

        # error for output level
        error_output = np.multiply(error, self.sigmoid_prime(prediction))
        # error generated by hidden level
        temp = np.dot(self.hidden_weights, error_output)
        error_hidden = np.multiply(temp, self.sigmoid_prime(self.first_layer_res))

        # 1d array dot product is not pretty
        delta_output_weight = np.dot(self.first_layer_res[:, None], error_output[None, :])
        delta_hidden_weight = np.dot(cur_x[:, None], error_hidden[None, :])

        # update weights using current level output error
        self.hidden_weights += np.multiply(self.learning_rate, delta_output_weight)
        self.weights += np.multiply(self.learning_rate, delta_hidden_weight)

        # update bias
        # self.bias += np.multiply(self.learning_rate, error_hidden[None, :])

    def sigmoid(self, score):
        return 1.0 / (1 + np.exp(-score))

    def sigmoid_prime(self, score):
        """
            derivative of sigmoid
        """
        return self.sigmoid(score) * (1 - self.sigmoid(score))  # by math formula

    def verify(self, test_data, test_labels, file_names):
        predictions = []
        error_count = 0

        bias = np.ones((test_data.shape[0], 1))
        test_data = np.concatenate((test_data, bias), axis=1)

        for image_data, image_res, file_name in zip(test_data, test_labels, file_names):
            pred = self.feed_forward(image_data)
            pred_label = 0
            if pred > 0.5:
                pred_label = 1
            # cur_res = "{} prediction: {}".format(file_name, pred_label)
            cur_res = "{} prediction: {}, result: {}".format(file_name, pred_label, pred)

            predictions.append(cur_res)

            if pred_label != image_res:
                error_count += 1

        # calc error rate
        accuracy = 1 - error_count * 1.0 / len(file_names)
        print('error_count', error_count)
        for p in predictions:
            print(p)
        print('\nAccuracy: ', accuracy)

        return predictions, accuracy


def bp_ffnn():
    np.random.seed(7777777)  # fix result by giving a random seed

    # get training and testing data
    X, y, train_file_names = get_training_set('downgesture_train.list')
    test_data, test_labels, test_file_names = get_training_set(
        'downgesture_test.list')  # training the model and prediction test data.
    start = time.clock()  # see how long it takes to run this code
    ffnn = FeedForwardNeuralNetwork(input_size=X[0].shape[0])
    ffnn.train_network(X, y)
    time_cost = time.clock() - start
    print('Training in Finished. takes {} sec\n\n'.format(time_cost))
    ffnn.verify(test_data, test_labels, test_file_names)

    # plt.plot(np.arange(0, 1000), error_list)  # print the error list
    # plt.show()


bp_ffnn()
