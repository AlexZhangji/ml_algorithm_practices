
import math
import random
import sys

import matplotlib.pyplot as plt
import pandas as pd


def get_data_points(fname):
    data_points_list = []
    with open(fname) as f:
        for line in f.readlines():
            x, y = line.replace(' ', '').split(',')[:2]
            data_points_list.append((float(x), float(y)))
    return data_points_list


def dp_distance(p1, p2):
    """
        get distance from two points,
        points in format of (x,y)
    """
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    return dist


def closest_k_cate_num(dp, centroids):
    """
        given a single data point and k list,
        return the category number the dp is closest to (in this case, either 0,1 or 2)
    """
    min_dist = sys.maxsize
    cate_num = -1

    for i, center in enumerate(centroids):
        cur_dist = dp_distance(dp, center)
        if cur_dist < min_dist:
            min_dist = cur_dist
            cate_num = i

    return cate_num


def label_data_by_distance(centroids, data_dict):
    """
        return dictionary of data labeled with given centroid list
        key as category number, values as list of data points
    """
    # init dictionary with keys as [0,1,2 ..., k], and val as empty list
    new_data_dict = {k: [] for k in range(len(centroids))}

    for key in data_dict.keys():
        for dp in data_dict[key]:
            # find and set category number for the data points
            cate_num = closest_k_cate_num(dp, centroids)
            new_data_dict[cate_num].append(dp)

    return new_data_dict


def get_new_centroid(data_dict, centroid_list):
    """
        given labled data
        return list of updated centroid, calc by average of all clustered data
    """
    centroid_count = len(centroid_list)
    new_centroid_list = []

    for i in range(centroid_count):
        dp_list = data_dict[i]

        if len(dp_list) != 0:  # calc the x and y coordinate for the new centroid with clustered data
            x_cord = sum([dp_[0] for dp_ in dp_list]) / float(len(dp_list))  # add float in case sum can be integer
            y_cord = sum([dp_[1] for dp_ in dp_list]) / float(len(dp_list))

            new_centroid_list.append((x_cord, y_cord))

    # centroid are replaced with same index, in case the cate number is mixed
    return new_centroid_list


def calc_variance_sum(data_dict, centroid_list):
    """
        sum of square distance between each data point and its assigned centroid
        use as evaluation for the goodness of the clustering
    """
    sum_sq_dist = 0
    for centroid, key in zip(centroid_list, data_dict.keys()):
        for dp in data_dict[key]:
            sum_sq_dist += pow(dp_distance(centroid, dp), 2)
    return sum_sq_dist


def k_means(data, k=3):
    """
        Return centroid for each clusters, and list of data points it has
    """
    centroid_list = []
    points_count = len(data)
    data_dict = {-1: data}  # convert data into dictionary, to keep the data type in next round

    # init k initial random centroid (on the data list)
    for _ in range(k):
        random_points = data[random.randint(0, points_count - 1)]
        centroid_list.append(random_points)

    # init the optimization val that k-means are minimizing
    variance_sum_first = sys.maxsize
    variance_sum_current = sys.maxsize - 1
    count = 0

    while variance_sum_first > variance_sum_current and count < 100:
        # categorize each by its distance to the centroid
        data_dict = label_data_by_distance(centroid_list, data_dict)
        # create new centroid by categorized data
        centroid_list = get_new_centroid(data_dict, centroid_list)

        # update the variance sum
        variance_sum_first = variance_sum_current
        variance_sum_current = calc_variance_sum(data_dict, centroid_list)

        count += 1

    return centroid_list, variance_sum_current, data_dict


def plot(centroids, fin_res):
    """
        given the final clustering result and centroids,
        plot the clustering with different color.
    """
    # original data points
    df = pd.read_csv('clusters.txt', header=None)
    df.columns = ['x', 'y']

    centers = pd.DataFrame(centroids)
    centers.columns = ['x', 'y']

    # plot clustering with diff color
    color_list = ['purple', 'green', 'blue']
    for i in range(3):
        cur_data = pd.DataFrame(fin_res[i])
        cur_data.columns = ['x', 'y']
        plt.scatter(cur_data['x'], cur_data['y'], s=77, c=color_list[i], marker='x')

    # plot the centroids with coordinates text
    plt.scatter([x[0] for x in centroids], [x[1] for x in centroids], s=177, color='red')
    for a, b in zip([x[0] for x in centroids], [x[1] for x in centroids]):
        plt.text(a, b, 'x:{}\ny:{}'.format(a, b), fontsize=15)

    plt.show()


def kmean_assignment():
    fname = 'clusters.txt'
    data_list = get_data_points(fname)  # list of DataPoint objects, in form of (1.23, -2.11)
    k = 3  # given K number in this case

    fin_centroids_res = []
    min_variance = sys.maxsize
    fin_data_dict = dict()

    for i in range(50):  # restart 50 times for smallest variance, and use the min variance centroids as result
        result, variance, res_data_dict = k_means(data_list, k)
        if variance < min_variance:
            min_variance = variance
            fin_centroids_res = result
            fin_data_dict = res_data_dict
    print('final result:{}\nwith variance:{}'.format(fin_centroids_res, min_variance))
    # print('data:{}'.format(fin_data_dict))
    return fin_centroids_res, fin_data_dict


# uncomment below to test k-means, comment out below to test gmm
fin_centroids_res, fin_cluster = kmean_assignment()
plot(fin_centroids_res, fin_cluster)
