import numpy as np
import pandas as pd
from scipy.spatial import distance

from KNeighborsClassifier import KNeighborsClassifier
from knn import Knn


def find_cluster(data_frame, data_frame_cluster):
    cluster = {}
    for i, center in enumerate(data_frame_cluster.values):
        cluster[i] = []

    for j, point in enumerate(data_frame.values):
        euclDist = float('inf')
        euclCenter = 0
        for i, center in enumerate(data_frame_cluster.values):
            dist = distance.euclidean(point, center)
            if dist < euclDist:
                euclDist = dist
                euclCenter = i

        # cluster[euclCenter] = []
        if cluster[euclCenter]:
            cluster[euclCenter].append(point)
        else:
            cluster[euclCenter] = [point]
    # print(cluster)
    return cluster


def mykmeans(X, k):
    # DONE create k random centroid from dataset.
    # TODO count euclidean distance from each centroid.
    # TODO we find the new centroid by taking the average of all the points assigned to that cluster.
    # TODO we repeat step 2 and 3 until none of the cluster assignments change. That means until our clusters remain stable, we repeat the algorithm
    try:
        data_frame = pd.DataFrame(data=X)
        # data_frame = data_frame
        data_frame_cluster = data_frame.sample(n=k)
        # print(data_frame_cluster)

        prev_centers = []
        while True:
            # Group data in clusters
            cluster = find_cluster(data_frame, data_frame_cluster)

            # Calculate new centroid
            centers = []
            for clusterKey, clusterValue in cluster.items():
                df = pd.DataFrame(clusterValue)
                center = []
                for column in df:
                    center.append(df[column].mean())
                centers.append(center)

            # Breaking condition, if prev centers and current centers are same
            if prev_centers == centers:
                break
            data_frame_cluster = pd.DataFrame(centers)
            prev_centers = centers
        print("For k = " + str(k) + " centers are: ")
        print(centers)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    data = np.genfromtxt("NBAstats.csv", delimiter=",", skip_header=1, usecols=(
        1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28))
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
    i = 0
    for column in data:
        for j in range(len(data[column])):
            data[column][j] = (data[column][j] - list_of_columns_mean[i]) / list_of_columns_standard_deviation[i]
        i += 1
    # knn_data = data.append(column_name)
    # knn_data = np.column_stack((data[:, 0].reshape(475, 1), k_means_data))
    # training_label = knn_data[0:375, 0].reshape(375, 1)
    # training_data = data[0:375, 1:]
    # test_label = knn_data[375:, 0].reshape(100, 1)
    # test_data = knn_data[375:, 1:]

    # mykmeans(X=data, k=3)
    # mykmeans(X=data, k=5)
    # data = pd.DataFrame(data=data, columns=[10, 13, 17, 20, 21, 22, 23])
    # mykmeans(data, 3)
    # mykmeans(data, 5)

    training_data = data[:-100]
    test_data = data.tail(100)

    classifier = Knn()
    predictions = []
    k = 3
    for x in range(len(test_data)):
        neighbors = classifier.get_nearest_neighbors(training_data.values, test_data.values[x], k)
        result = classifier.get_response(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(test_data[x][-1]))
    accuracy = classifier.get_accuracy(test_data.values, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')

    # classifier = KNeighborsClassifier()
    # classifier.fit(training_data, column_name)
    # print("For KNN Classifier-")
    # print("Predictions on Test set:\n{}".format(classifier.predict(test_data)))
    # print("Accuracy of Test set: {:.2f}".format(classifier.score(test_data, column_name)))
