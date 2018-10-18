import numpy as np
import pandas as pd
from scipy.spatial import distance

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
