import numpy as np
import pandas as pd
from scipy.spatial import distance


def mykmeans(X, k):
    pass
    # # DONE create k random centroid from dataset.
    # # TODO count euclidean distance from each centroid.
    # # TODO we find the new centroid by taking the average of all the points assigned to that cluster.
    # # TODO we repeat step 2 and 3 until none of the cluster assignments change. That means until our clusters remain stable, we repeat the algorithm
    # try:
    #     data_frame = pd.DataFrame(data=X)
    #     data_frame_cluster = data_frame.sample(n=k)
    #     mean_of_cluster = [cluster.mean() for cluster in data_frame_cluster.values]
    #     d = {}
    #     for i, cluster in enumerate(data_frame.values):
    #         d[i] = []
    #         for j, point in enumerate(data_frame.values):
    #             d[cluster.mean()].append(distance.euclidean(point, cluster))
    #
    #     # calculate difference taking mean of every cluster and subtracting it from each mean.
    #     # which one has lowest distance we can say that belongs to that cluster.
    #     for i in mean_of_cluster:
    #         for j in d:
    #             difference.append(i - j)
    #     # difference
    # except Exception as e:
    #     print(e)


if __name__ == '__main__':
    data = np.genfromtxt("NBAstats.csv", delimiter=",", skip_header=1, usecols=(1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28))
    # k_means_data = data[:, 1:]
    knn_data = pd.read_csv("NBAstats.csv", delimiter=",")
    data = pd.DataFrame(data=data)
    column_name = knn_data.columns.values
    transpose_data_frame = pd.DataFrame(column_name).transpose()
    column_name = transpose_data_frame.drop(labels=[0, 1, 3], axis=1)
    data = data.drop(0, 1)
    list_of_columns_mean = []
    list_of_columns_standard_deviation = []
    for column in data:
        list_of_columns_mean.append(data[column].mean())
        list_of_columns_standard_deviation.append(data[column].std())
    # standard_deviation = np.std(k_means_data)
    i = 0
    for column in data:
        for j in range(len(data[column])):
            data[column][j] = (data[column][j] - list_of_columns_mean[i]) / list_of_columns_standard_deviation[i]
        i += 1
    # knn_data = data.append(column_name)
    # mean_of_data = np.mean(k_means_data)
    # k_means_data = (k_means_data - mean_of_data) / standard_deviation
    # knn_data = np.column_stack((data[:, 0].reshape(475, 1), k_means_data))
    # training_label = knn_data[0:375, 0].reshape(375, 1)
    # training_data = data[0:375, 1:]
    # test_label = knn_data[375:, 0].reshape(100, 1)
    # test_data = knn_data[375:, 1:]
    mykmeans(X=data, k=3)
    mykmeans(X=data, k=5)
