import numpy as np
import pandas as pd
from scipy.spatial import distance


def mykmeans(X, k):
    # DONE create k random centroid from dataset.
    # TODO count euclidean distance from each centroid.
    # TODO we find the new centroid by taking the average of all the points assigned to that cluster.
    # TODO we repeat step 2 and 3 until none of the cluster assignments change. That means until our clusters remain stable, we repeat the algorithm
    try:
        data_frame = pd.DataFrame(data=X)
        data_frame_cluster = data_frame.sample(n=k)
        mean_of_cluster = [cluster.mean() for cluster in data_frame_cluster.values]
        d = {}
        # d = []
        for i, cluster in enumerate(data_frame_cluster.values):
            d[cluster.mean()] = []
            for j, point in enumerate(data_frame.values):
                d[cluster.mean()].append(distance.euclidean(point, cluster))
                # d.append(distance.euclidean(point, cluster))
        difference = []
        print(d)
        # calculate difference taking mean of every cluster and subtracting it from each mean.
        # which one has lowest distance we can say that belongs to that cluster.
        for i in mean_of_cluster:
            for j in d:
                difference.append(i-j)
        # difference
    except Exception as e:
        print(e)


if __name__ == '__main__':
    data = np.genfromtxt("NBAstats.csv", delimiter=",", skip_header=1, usecols=(1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28))
    k_means_data = data[:, 1:]
    # data = pd.read_csv("NBAstats.csv", delimiter=",")
    standard_deviation = np.std(k_means_data)
    mean_of_data = np.mean(k_means_data)
    k_means_data = (k_means_data - mean_of_data) / standard_deviation
    knn_data = np.column_stack((data[:, 0].reshape(475, 1), k_means_data))
    training_label = knn_data[0:375, 0].reshape(375, 1)
    training_data = data[0:375, 1:]
    test_label = knn_data[375:, 0].reshape(100, 1)
    test_data = knn_data[375:, 1:]
    mykmeans(X=k_means_data, k=3)
