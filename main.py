from Kmeans import mykmeans
import pandas as pd
import numpy as np
from copy import deepcopy


def main():
    try:
        data = np.genfromtxt("NBAstats.csv", delimiter=",", skip_header=1, usecols=(
            1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28))
        knn_data = pd.read_csv("NBAstats.csv", delimiter=",")
        data = pd.DataFrame(data=data)
        position_of_player = knn_data['Pos']
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
        kmeans_data = deepcopy(data)
        data.insert(0, column=0, value=position_of_player)
        mykmeans(X=kmeans_data, k=3)
        mykmeans(X=kmeans_data, k=5)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
