import random
import numpy as np
import pandas as pd
import math
import sys

IntegerTypes = (int)
StringTypes = (str)

class Agglomerative:
    __centroid = []
    __distance = []

    def __init__(self, number_of_cluster, linkage) :
        if not isinstance(number_of_cluster, IntegerTypes) :
            raise TypeError('number_of_cluster must be a integer')
        elif not isinstance(linkage, StringTypes) :
            raise TypeError('linkage must be a integer')

        self.__number_of_cluster = number_of_cluster
        self.__linkage = linkage

    # def __euclidean(self, X, y) :
    #     return np.linalg.norm(X - y)

    def fit(self, X) :
        # X is pandas.dataframe
        
        # raise error
        if not isinstance(X, pd.core.frame.DataFrame) :
            raise TypeError("X must be a pandas.core.frame.DataFrame")

        self.__distance = pairwise_distances(X, metric='euclidean')
        self.__X_train = X

        for iterate in range(len(self.__distance)):
            min_distance = self.__distance[1][0]
            min_P1 = 1
            min_P2 = 0
            temp_centroid = []

            for i in range(len(self.distance)):
                temp_centroid.append(i)

            self.__centroid[0] = temp_centroid.copy()

            for i in range(len(self.__distance)):
                for j in range(i):
                    if min_distance > self.distance[i][j]:
                        min_distance = self.distance[i][j]
                        min_P1 = i
                        min_P2 = j

            if (self.linkage == "single"):
                # do single linkage agglomerative here

                for i in range(len(self.distance)):
                    if i != min_P1:
                        self.distance[i][min_P1] = min(self.distance[i][min_P1], self.distance[i][min_P2])
                        self.distance[min_P1][i] = min(self.distance[i][min_P1], self.distance[i][min_P2])

            elif (self.linkage == "complete"):
                # do complete linkage agglomerative here

                for i in range(len(self.distance)):
                    if i != min_P1:
                        self.distance[i][min_P1] = max(self.distance[i][min_P1], self.distance[i][min_P2])
                        self.distance[min_P1][i] = max(self.distance[i][min_P1], self.distance[i][min_P2])

            elif (self.linkage == "average"):
                # do average linkage agglomerative here

                for i in range(len(self.distance)):
                    if i != min_P1:
                        self.distance[i][min_P1] = (self.distance[i][min_P1] + self.distance[i][min_P2])/2
                        self.distance[min_P1][i] = (self.distance[i][min_P1] + self.distance[i][min_P2])/2

            elif (self.linkage == "average-group" or self.linkage == "ward"):
                # do average-group linkage agglomerative here

            for i in range (0,input.shape[0]):
                input[row_index][i] = sys.maxsize
                input[i][row_index] = sys.maxsize
                  
            for i in range(len(temp_centroid)):
                if(temp_centroid[i] == max(min_P1, min_P2)):
                    temp_centroid[i] = min(min_P1, min_P2)

            self.__centroid[iterate] = temp_centroid.copy()

    def predict(self, X) :
        # X is pandas.dataframe

        if not isinstance(X, pd.core.frame.DataFrame) :
            raise TypeError("X must be a pandas.core.frame.DataFrame")
        elif X.select_dtypes(exclude=['number']).empty :
            raise TypeError("X must be all number")

        self.__batch_X = X
        self.__forward_pass()
        result = list(map(lambda x: x[1 + self.__hidden_layer][0], self.__outputs))

        return result
