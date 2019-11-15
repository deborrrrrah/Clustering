import random
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics.pairwise import pairwise_distances
import math
import sys


IntegerTypes = (int)
StringTypes = (str)

class Agglomerative:
    __clusters = {}
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

        temp_centroid = []

        for i in range(len(self.__distance)):
            temp_centroid.append(i)

        self.__clusters[0] = temp_centroid.copy()

        for iterate in range(1, len(self.__distance)):
            min_distance = self.__distance[1][0]
            min_P1 = 1
            min_P2 = 0

            for i in range(len(self.__distance)):
                for j in range(i):
                    if min_distance > self.__distance[i][j]:
                        min_distance = self.__distance[i][j]
                        min_P1 = i
                        min_P2 = j

            if (self.linkage == "single"):
                # do single linkage agglomerative here

                for i in range(len(self.__distance)):
                    if i != min_P1:
                        distance_change = min(self.__distance[i][min_P1], self.__distance[i][min_P2])
                        self.__distance[i][min_P1] = distance_change
                        self.__distance[min_P1][i] = distance_change

            elif (self.linkage == "complete"):
                # do complete linkage agglomerative here

                for i in range(len(self.__distance)):
                    if i != min_P1:
                        distance_change = max(self.__distance[i][min_P1], self.__distance[i][min_P2])
                        self.__distance[i][min_P1] = distance_change
                        self.__distance[min_P1][i] = distance_change

            elif (self.linkage == "average-group"):
                # do average linkage agglomerative here

                for i in range(len(self.__distance)):
                    if (i != min_P1 and i != min_P2):
                        distance_change = (self.__distance[i][min_P1] + self.__distance[i][min_P2])/2
                        self.__distance[i][min_P1] = distance_change
                        self.__distance[min_P1][i] = distance_change

            elif (self.linkage == "average"):
                # do average-group linkage agglomerative here
                
                for i in range(len(self.__distance)):
                    if(i != min_P1 and i != min_P2):
                        if (i in self.__clusters[iterate-1]) and (min_P1 in self.__clusters[iterate-1]):
                            indices_0 = [o for o, u in enumerate(self.__clusters[iterate-1]) if u == i]
                            indices_1 = [o for o, u in enumerate(self.__clusters[iterate-1]) if u == min_P1]

                            sumDist = 0;
                            for index1 in indices_1:
                                for index2 in indices_0:
                                    sumDist += distance.euclidean(data[index1],data[index2])
                            meanDist = sumDist / (len(indices_1) + len(indices_0))
                            self.__distance[min_P1][i] = meanDist;
                            self.__distance[i][min_P1] = meanDist;

            for i in range(len(self.__distance)):
                self.__distance[min_P1][i] = sys.maxsize
                self.__distance[i][min_P1] = sys.maxsize
                  
            for i in range(len(temp_centroid)):
                if(temp_centroid[i] == max(min_P1, min_P2)):
                    temp_centroid[i] = min(min_P1, min_P2)

            self.__clusters[iterate] = temp_centroid.copy()

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
