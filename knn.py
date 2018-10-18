from scipy.spatial import distance

class Knn(object):

    @classmethod
    def get_nearest_neighbors(cls, training_dataset, test_data, k):

        distances = []
        for training_data in training_dataset:
            euclidean_dist = distance.euclidean(training_data[1:], test_data[1:])
            distances.append((euclidean_dist, training_data))
        
        # Sort by distances
        sorted_distances = sorted(euclidean_dist, key=lambda x: x[0])

        return [distance_data[1] for distance_data in sorted_distances[:k]]

    @classmethod
    def get_classification(cls, neighbours):
        """ Returns label
        """
        class_votes = dict()
        for neighbour in neighbours:
            label = neighbour[0]
            vote = class_votes.get(label,0)
            class_votes[label] = vote + 1
            
        sorted_votes = sorted(
            list(class_votes.items()), key=lambda x: x[0], reverse=True)
        return sorted_votes[0][0]

    @classmethod
    def get_accuracy(cls, prediction_n_test):
        correct = 0
        for prediction, test in prediction_n_test:
            if prediction == test:
                correct += 1
        return (float(correct)/float(len(prediction_n_test))) * 100.0
    

def calculate_knn_and_accuracy(training_data, test_data, k_val):

    # Get list of nearest neighbors for each test
    prediction_n_test_list = list()
    for test_data_instance in test_data:
        nearest_neighbors = Knn.get_nearest_neighbors(
            training_dataset=training_data, 
            test_data=test_data_instance,
            k=k_val)
        calculated_classification = Knn.get_classification(nearest_neighbors)
        prediction_n_test_list.append((test_data_instance[0], calculated_classification)) # Given Classification, Calculated Classification

    # Calculate Accuracy
    return Knn.get_accuracy(prediction_n_test_list)


# Get Data
def excute(master_dataset, k_values):

    master_data = master_dataset
    training_data = master_data[:-100]
    test_data = master_data[-100:]

    k_values = [1, 5, 10, 30]

    for k_val in k_values:
        accuracy = calculate_knn_and_accuracy(training_data, test_data, k_val)
        print("K:",k_val, "Accuracy:", accuracy)
