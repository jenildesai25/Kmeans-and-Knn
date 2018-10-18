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

    @classmethod
    def get_response(cls, neighbours):
        class_votes = {}
        for x in range(len(neighbours)):
            response = neighbours[x][-1]
            if response in class_votes:
                class_votes[response] += 1
            else:
                class_votes[response] = 1
        sorted_votes = sorted(class_votes.iteritems(), key=lambda x: x[0], reverse=True)
        return sorted_votes[0][0]

    @classmethod
    def get_accuracy(cls, test_set, predictions):
        correct = 0
        for x in range(len(test_set)):
            if test_set[x][-1] is predictions[x]:
                correct += 1
        return (correct/float(len(test_set))) * 100.0