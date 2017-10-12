import random

import matplotlib.pyplot as plt
import numpy as np

# set precision
np.set_printoptions(precision=3)


def get_dis_matrix(fname='fastmap-data.txt'):
    """
        return a numpy array for distance,
        array[i][j] will return distance between point i and j
    """
    distance_matrix = np.zeros((11, 11))  # init a 11 by 11 matrix to fill

    with open(fname) as f:
        for line in f.readlines():
            x, y, cur_dist = [int(num) for num in line.split('\t')]

            distance_matrix[x][y] = cur_dist
            distance_matrix[y][x] = cur_dist
    return distance_matrix


def get_furthest_point(matrix, point):
    """
        given a matrix, a point, return its furthest pair of point
        O(n) runtime
    """
    # furthest_p = -1
    max_dist = max(matrix[point,])
    for i, val in enumerate(matrix[point,]):
        if val == max_dist:
            return i
    return -1


def get_far_pair(dis_matrix):
    """
        use a random point, find its furthest pair, and then, find furthest pair for that point
        return the final pair and distance
    """
    rand_p = random.randint(1, 10)

    # find furthest pair for rand_p
    furthest_p1 = get_furthest_point(dis_matrix, rand_p)
    furthest_p2 = get_furthest_point(dis_matrix, furthest_p1)

    for i in range(3):  # run 3 more times to make sure we get the fairest pair
        furthest_p1 = get_furthest_point(dis_matrix, furthest_p2)
        furthest_p2 = get_furthest_point(dis_matrix, furthest_p1)

    max_dist = dis_matrix[furthest_p1][furthest_p2]

    # make sure using the smaller one as the zero point
    p1 = min([furthest_p1, furthest_p2])
    p2 = max([furthest_p1, furthest_p2])
    # print('p1:{}\np2:{}\nmax:{}'.format(p1, p2, max_dist))
    return p1, p2, max_dist


def get_projected_dist(o, a, b, matrix):
    """
        get the projected distance using piv_a and piv_b to form line (base of triangle)
    """
    temp = matrix[a][b] ** 2 + matrix[a][o] ** 2 - matrix[o][b] ** 2
    denominator = 2 * matrix[a][b]
    projected_dist = float(temp / denominator)
    # print('projected_dist for {} with({},{}) is :{}'.format(o, a, b, projected_dist))
    return projected_dist


def fastmap(dist_matrix, dim=2):
    result_matrix = np.zeros((11, 2))

    while dim > 0:
        # identify the far pair and distance by heuristic
        piv_a, piv_b, piv_dist = get_far_pair(dist_matrix)

        if dis_matrix[piv_a][piv_b] == 0:
            print('dist between two far pivots are zero')
            result_matrix[:, 2 - dim] = 0.0
        else:
            # find projected distance Xi for each data point
            num_list = [num for num in range(1, 11)]
            for num in num_list:
                proj_dist = get_projected_dist(num, piv_a, piv_b, dist_matrix)
                result_matrix[num][2 - dim] = proj_dist

        # print('result matrix:\n{}'.format(result_matrix))
        # update the dis matrix
        dist_matrix = update_dist_matrix(dist_matrix, result_matrix, dim)
        dim -= 1

    return result_matrix


def update_dist_matrix(dist_matrix, result_matrix, dim):
    """
        update the distant matrix with new distance in hyperplane
    """
    updated_dist_matrix = np.zeros((11, 11))  # probably should not put here but pass in.
    for i in range(1, 11):
        for j in range(1, 11):
            updated_dist_matrix[i, j] = np.sqrt(np.square((dist_matrix[i, j])) -
                                                np.square(result_matrix[i, 2 - dim] - result_matrix[j, 2 - dim]))
    # print(updated_dist_matrix)
    return updated_dist_matrix


def plotter(result_matrix, words_list):
    """
        plot the word list and result location
    """
    # print('word list:{}\n res len:{}'.format(len(words_list), len(result_matrix)))
    x_list, y_list = [dp[0] for dp in result_matrix], [dp[1] for dp in result_matrix]
    plt.scatter(x_list, y_list)
    for i, word in enumerate(words_list):
        plt.annotate(word, result_matrix[i])
    plt.show()


def get_labels(fname):
    with open(fname) as f:
        return f.readlines()


fname = 'fastmap-data.txt'
label_fname = 'fastmap-wordlist.txt'
dis_matrix = get_dis_matrix(fname)
label_list = get_labels(label_fname)

# run the fastmap algorithm
dim = 2
result = fastmap(dis_matrix, dim)[1:]  # slice first entry cuz it's zero.
print('\n=== final ===\n{}'.format(result))

plotter(result, label_list)
