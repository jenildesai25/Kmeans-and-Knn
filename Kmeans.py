import numpy as np
import pandas as pd
from scipy.spatial import distance


def mykmeans(X, k):
  pass
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
    mykmeans(X=data, k=3)
    mykmeans(X=data, k=5)