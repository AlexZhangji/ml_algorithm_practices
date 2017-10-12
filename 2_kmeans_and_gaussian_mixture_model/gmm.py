
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
import kmeans


def get_data_points(fname):
    data_points_list = []
    with open(fname) as f:
        for line in f.readlines():
            x, y = line.replace(' ', '').split(',')[:2]
            data_points_list.append((float(x), float(y)))
    return data_points_list


def init_gaussians(data, k=3):
    """ return a gaussian dictionary that consists of means. cov and size. """
    gaussians = {i: {} for i in range(k)}  # init gaussian model list as {0:{'mean':12.0, cov:{}}}

    # use KMeans algo to init the means and clusters
    mean_dp, clusters = kmeans.kmean_assignment()

    for i in range(k):
        gaussians[i]['mean'] = np.mean(clusters[i], axis=0)
        gaussians[i]['prob'] = len(clusters[i]) / len(data)

        # calc coveriance
        x_vals = [dp[0] for dp in clusters[i]]
        y_vals = [dp[1] for dp in clusters[i]]
        gaussians[i]['cov'] = np.cov(x_vals, y_vals, bias=True)

        print('Gaussian Model')
        print('mean:{}'.format(gaussians[i]['mean']))
        print('prob:{}'.format(gaussians[i]['prob']))
        print('cov:{}\n'.format(gaussians[i]['cov']))

    return gaussians


def init_random(data, k=3):
    gaussians = {i: {} for i in range(k)}  # init data dict as {0:{'mean':12.0, cov:{}}}

    x_vals = [dp[0] for dp in data]
    y_vals = [dp[1] for dp in data]
    # init same cov matrix for each cluster, using all data points
    cov = np.cov(x_vals, y_vals, bias=True)

    for i in range(k):
        gaussians[i]['mean'] = np.array(data[random.randint(0, len(data) - 1)])  # use random data points on the graph
        gaussians[i]['prob'] = 1 / 3  # random init asssume the prior prob the same for each cluster
        gaussians[i]['cov'] = cov

        # print('mean:{}'.format(gaussians[i]['mean']))
        # print('prob:{}'.format(gaussians[i]['prob']))
        # print('cov')
        # print(gaussians[i]['cov'])
        # print('')

    return gaussians


def expect(data, gauss_dict, k=3):
    """
        using data and gaussian models to calc the PDF for each model
        and using PDF to calc the each point's possibility to belong to certain cluster
    """
    data_count = len(data)
    # init possibility density function dictionary
    pdf_dict = {cluster_num: [dp for dp in range(data_count)] for cluster_num in range(k)}

    # calc pdf for each gaussian model
    for i in range(3):
        cur_gauss = gauss_dict[i]
        cov = cur_gauss['cov']

        # calc the normal factor of the equation : 1/sqrt((2*Pi*)^2*det(cov)
        det_cov = np.linalg.det(cov)
        normal_factor = 1 / np.sqrt((2 * np.pi) ** 2 * det_cov)

        # save for calc the index
        mean = cur_gauss['mean']
        inv_cov = np.linalg.inv(cov)

        # calc the index for e (-0.5*(x-u).T*inv_Cov*(x-u))
        for j in range(data_count):
            temp = data[j] - mean
            trans_temp = temp.T  # transposed temp (x-u).T
            temp_product = np.dot(-0.5, trans_temp)  # same as direct multiply when dot just number, but faster
            temp_product = np.dot(temp_product, inv_cov)
            temp_product = np.dot(temp_product, temp)
            # save the pdf for each points and cluster
            pdf_dict[i][j] = normal_factor * np.exp(temp_product)

    # similarly, calc the Ric for each points belong to every cluster
    ric_dict = {dp: [cluster_num for cluster_num in range(k)] for dp in range(data_count)}
    for i in range(data_count):
        total_pdf = 0.0

        # get the sum of weighted pdf
        for j in range(k):
            weighted_pdf = pdf_dict[j][i] * gauss_dict[j]['prob']  # get pdf for current dp
            total_pdf += weighted_pdf
        for j in range(k):
            ric_dict[i][j] = gauss_dict[j]['prob'] * pdf_dict[j][i] / total_pdf  # assign Ric for each point by cluster

    return ric_dict, pdf_dict


def maximization(ric_dict, data, k=3):
    """
        given Ric, update gaussian model's params
        mean, cov_matrix and prob
     """
    # init data dict as {0:{'mean':12.0, cov:{}}}, need Mean, Prob and Cov matrix
    new_gaussian = {i: {} for i in range(k)}
    data_count = len(data)

    for j in range(k):
        # new mean
        sum_ric = 0.0
        sum_weighted_x = 0.0
        sum_weighted_y = 0.0

        for i in range(data_count):
            sum_ric += ric_dict[i][j]
            sum_weighted_x += ric_dict[i][j] * data[i][0]
            sum_weighted_y += ric_dict[i][j] * data[i][1]
        # new mean
        new_gaussian[j]['mean'] = np.divide(np.array([sum_weighted_x, sum_weighted_y]), sum_ric)

        # new prob
        new_gaussian[j]['prob'] = sum_ric / data_count

        # new cov matrix, avoid singular matrix
        weighted_cov = np.zeros((2, 2))
        for i in range(data_count):
            temp = data[i] - new_gaussian[j]['mean']
            product_temp = np.dot(temp.T.reshape(2, 1), temp.reshape(1, 2))
            product_temp = product_temp * ric_dict[i][j]
            weighted_cov = np.add(weighted_cov, product_temp)

        new_gaussian[j]['cov'] = weighted_cov / sum_ric

    return new_gaussian


def calc_likelihood(pdf, gaussians, data, k_num):
    likelihood = 0.0
    for i in range(len(data)):
        sum_cluster_pdf = 0.0
        for j in range(0, k_num):
            sum_cluster_pdf += np.multiply(pdf[j][i], gaussians[j]['prob'])

        likelihood += np.log(sum_cluster_pdf)

    # print('lle: {}'.format(likelihood))
    return likelihood


def gmm(data, k=3, threshold=0.0001):
    # init clusters with random selected data points,
    gaussians = init_gaussians(data)  # init with means, cov and size
    # gaussians = init_random(data)  # init with random data points and 1/3 prob and cov for whole data set.

    iter_count = 0
    # calc Ric for current data
    weighted_prob_dict, pdf_dict = expect(data, gaussians, k)
    likelyhood = calc_likelihood(pdf_dict, gaussians, data, k)
    new_likelyhood = sys.maxsize

    # iterate until converge
    while abs(likelyhood - new_likelyhood) > threshold:
        iter_count += 1
        # print('run {} times'.format(iter_count))
        gaussians = maximization(weighted_prob_dict, data, k)
        weighted_prob_dict, pdf_dict = expect(data, gaussians, k)

        # update values
        likelyhood = new_likelyhood
        new_likelyhood = calc_likelihood(pdf_dict, gaussians, data, k)

    centroids = [gauss['mean'] for gauss in gaussians.values()]
    prob = [gauss['prob'] for gauss in gaussians.values()]
    cov = [gauss['cov'] for gauss in gaussians.values()]
    print(
        '\n\n==== one iter ===\nrun {} times\ncentroids:{}\nprob:{}\ncov:{}\n'.format(iter_count, centroids, prob, cov))
    print('liklihood:{}'.format(new_likelyhood))
    return centroids, prob, cov, new_likelyhood


def plot(centroids, cov_list):
    """
        for plotting the ellipse with cov and cetroids,
        not written by me, changed from demo in matplotlib offical site
    """
    for i in range(len(cov_list)):
        cov = cov_list[i]
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)

        ax = plt.subplot(111, aspect='equal')
        for j in range(1, 4):
            ell = Ellipse(xy=centroids[i],
                          width=lambda_[0] * j * 2, height=lambda_[1] * j * 2,
                          angle=np.rad2deg(np.arccos(v[0, 0])))
        ell.set_facecolor('none')
        ax.add_artist(ell)
        # plt.scatter(x, y)

    # plot all pd
    df = pd.read_csv('clusters.txt', header=None)
    df.columns = ['x', 'y']
    plt.scatter(df['x'], df['y'])

    plt.show()


fname = 'clusters.txt'
data_list = get_data_points(fname)  # list of DataPoint objects
k = 3  # given K number in this case

max_lle = -sys.maxsize
centroids, prob, cov = None, None, None
lle_list = []

for i in range(1):  # if using k means, only need one iteration, cuz its fixed for this initialization
    # for i in range(20):  # if using random, run sevral times, and use the one has highest log likelihood
    new_centroids, new_prob, new_cov, new_lle = gmm(data_list, k)
    lle_list.append(new_lle)
    if new_lle > max_lle:
        max_lle = new_lle
        centroids, prob, cov = new_centroids, new_prob, new_cov

print('\n\n==== final===\nCentroids:{}\nAmplitude:{}\nCov Matrix:{}\n'.format(centroids, prob, cov))
plot(centroids, cov)
