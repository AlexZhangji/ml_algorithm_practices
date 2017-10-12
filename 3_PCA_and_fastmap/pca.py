import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # need to keep this for 3D plot

# from mpl_toolkits.mplot3d import Axes3D  # need to keep this for 3D plot


def get_data(fname):
    res_data = []
    with open(fname) as f:
        for line in f.readlines():
            cur_data = [float(dp.strip()) for dp in line.split('\t')]
            res_data.append(cur_data)

    return np.array(res_data)


def get_mean_vec(dps):
    """
        return mean vector for each columns
    """
    mean_vec_ = []
    for i in range(len(dps[0])):
        data_cols = [dp[i] for dp in dps]
        dp_row_mean = np.mean(data_cols)
        mean_vec_.append(dp_row_mean)
    return np.array(mean_vec_)


def normalize(dps, mean_v):
    """
        get normalized data,
        for all dps, Xi => Xi - mean

        and turns out, this whole function can be simplified as:
        dps_matrix - mean_v
        numpy is powerful...
    """
    normalized_data = []

    for i in range(len(dps[0])):
        data_cols = [dp[i] for dp in dps]
        cur_data = [dp - mean_v[i] for dp in data_cols]

        normalized_data.append(cur_data)
    return np.array(normalized_data)


def pca(data_points):
    # pre-processing, normalize data
    mean_vec = get_mean_vec(data_points)  # get mean vec for each column
    normalized_dp = normalize(data_points, mean_vec)  # this whole function can be simplified as: dps_matrix - mean_v

    # calc the cov matrix
    cov_matrix = np.cov(normalized_dp)

    # calc the eig vectors
    eig_vals, eig_vec = np.linalg.eig(cov_matrix)  # again, all hail Numpy :D

    # sort eigenvalues in descending order
    # make a (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vec[:, i]) for i in range(len(eig_vals))]

    # sort the (eigenvalue, eigenvector) tuples from high to low by eigenvalues
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    result_matrix = np.hstack([eig_vec[1].reshape(3, 1) for eig_vec in eig_pairs[:2]])

    return result_matrix


def plotter(matrix, data):
    """
        not sure how should this been done...
        not pretty, but works
    """
    # plot 3d data points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = [data[:, i] for i in range(3)]
    ax.scatter(xs, ys, zs, color='red', marker='^')

    # plot the transformed 2D data
    transformed = matrix.T.dot(data.T)
    plt.plot(transformed[0, :], transformed[1, :], 'o', markersize=7, color='blue', alpha=0.5, label='class1')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.legend()
    plt.title('3D ordinal data vs 2D transformed ones ')
    plt.show()


fname = 'pca-data.txt'
data_points = get_data(fname)
result_matrix = pca(data_points)
print(result_matrix)
""" 
    uncomment the plotter part to show the visualization for 3D vs 2D points
    Note that [from mpl_toolkits.mplot3d import Axes3D] need to be in import lines for 3D plot
    this line somehow will get automatically deleted by Pycharm once you format code (Ctrl/Cmd+L)
"""
# plotter(result_matrix, data_points)
