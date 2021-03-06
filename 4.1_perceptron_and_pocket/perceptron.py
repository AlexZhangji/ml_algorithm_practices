import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D  # need to keep this for 3D plot


# from mpl_toolkits.mplot3d import Axes3D  # need to keep this for 3D plot


def get_data(fname='classification.txt'):
    """
        given a file name, return a pandas data frame
        assume separated by comma
    """
    data_raw = np.array(pd.read_csv(fname, sep=",", header=None))
    X, y = data_raw[:, :3], data_raw[:, 3]
    return X, y


class Perceptron:
    def __init__(self, X, y, max_iter=77777, learning_rate=1, target_accuracy=1.0):
        self.X = X
        self.y = y
        self.weights = np.ones(self.X.shape[1] + 1)  # init weights as dimension + 1, for ease of matrix calc
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.target_accuracy = target_accuracy

    def fit(self):
        x_zero = np.ones((self.X.shape[0], 1))  # bias value add to X.
        self.X = np.concatenate((x_zero, self.X), axis=1)  # init X0 as 1

        iteration = 0
        while True:
            predictions = np.dot(self.weights, self.X.T)
            error_count = 0
            for pred, cur_x, cur_y in zip(predictions, self.X, self.y):
                if self.get_sign(pred) != cur_y:
                    error_count += 1
                    self.weights += self.learning_rate * cur_x * cur_y
                    break  # reevaluate the prediction with new weights

            iteration += 1

            if error_count == 0:
                print("finished with {} iteration.".format(iteration))
                break

            # if error_count / 2000.0 <= 1 - self.target_accuracy:
            #     print("finished with {} iteration.".format(iteration))
            #     break

            if iteration >= self.max_iter:
                print('max iteration reached')
                break

        return self.weights

    def get_sign(self, val):
        if val > 0:
            return 1
        else:
            return -1

    def verify(self):
        error_count = 0

        predictions = np.dot(self.weights, self.X.T)
        for pred, cur_x, cur_y in zip(predictions, self.X, self.y):
            if self.get_sign(pred) != cur_y:
                error_count += 1

        accuracy = error_count / 2000.0
        print('error rate: {} '.format(accuracy))
        return error_count


def plotter(X, y, weights):
    # plot 3d data points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    dp1, dp2 = [], []
    for dp, cur_y in zip(X, y):
        if cur_y == 1:
            dp1.append(dp)
        else:
            dp2.append(dp)
    dp1 = np.array(dp1)
    dp2 = np.array(dp2)

    # scatter for raw data
    ax.scatter(dp1[:, 0], dp1[:, 1], dp1[:, 2], color='red', marker='^')
    ax.scatter(dp2[:, 0], dp2[:, 1], dp2[:, 2], color='blue', marker='^')

    # plot the hyper plane by weights
    point = np.array([0, 0, 0])
    normal = np.array(weights[1:])
    d = -point.dot(normal)

    # create x,y
    xx, yy = np.meshgrid(range(2), range(2))
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    ax.plot_surface(xx, yy, z, alpha=0.2)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.legend()
    plt.title('Perceptron Algorithm\n3D data points with hyper plane generated by weights')
    plt.show()


start = time.clock()  # see how long it takes to run this code
X, y = get_data()
percep_algo = Perceptron(X, y)
result_weights = percep_algo.fit()
print('result weights:{}'.format(result_weights))
percep_algo.verify()
# print('error rate:{}'.format((percep_algo.verify()) / 2000.0))
print('takes {} sec'.format(time.clock() - start))
plotter(X, y, result_weights)
