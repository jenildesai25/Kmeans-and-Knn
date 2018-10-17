from scipy.spatial import distance

class Knn(object):

    @classmethod
    def get_nearest_neighbors(cls, training_dataset, test_data, k):

        distances = []
        for training_data in training_dataset:
            euclidean_dist = distance.euclidean(training_data, test_data)
            distances.append((euclidean_dist, training_data))
        
        # Sort by distances
        sorted_distances = sorted(euclidean_dist, key=lambda x: x[0])

        return [distance_data[1] for distance_data in sorted_distances[:k]]

