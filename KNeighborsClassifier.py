import numpy as np
from sortedcontainers import SortedList


class KNeighborsClassifier(object):
    k = 5

    def initialization(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        y = np.zeros(len(X))
        for i, x in enumerate(X):  # test points
            sorted_list = SortedList(self.k)  # stores (distance, class) tuples
            for j, xt in enumerate(self.X):  # training points
                difference = x - xt
                dotpro = difference.dot(difference)
                if len(sorted_list) < self.k:
                    sorted_list.add((dotpro, self.y[j]))
                else:
                    if dotpro < sorted_list[-1][0]:
                        del sorted_list[-1]
                        sorted_list.add((dotpro, self.y[j]))

            vote = {}
            for _, v in sorted_list:
                vote[v] = vote.get(v, 0) + 1

            max_vote = 0
            max_vote_class = -1
            for v, count in vote.items():
                if count > max_vote:
                    max_vote = count
                    max_vote_class = v
            y[i] = max_vote_class
        return y