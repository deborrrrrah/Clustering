import random
import numpy as np
import pandas as pd
from scipy.spatial import distance
import math
import sys
from copy import deepcopy
from sklearn.metrics import pairwise_distances


IntegerTypes = (int)
StringTypes = (str)
LinkageList = ['single', 'complete', 'average-group', 'average']

class Agglomerative:
    __clusters = {}

    def __init__(self, number_of_cluster, linkage) :
        if not isinstance(number_of_cluster, IntegerTypes) :
            raise TypeError('number_of_cluster must be a integer')
        elif not isinstance(linkage, StringTypes) :
            raise TypeError('linkage must be a string')
        elif isinstance(linkage, StringTypes) :
            if linkage not in LinkageList :
                raise TypeError('linkage must be either single, complete, average-group or average')
        self.__number_of_cluster = number_of_cluster
        self.__linkage = linkage

    @staticmethod
    def __euclidean(x, y) :
        if len(x) != len(y) :
            raise Exception('length x and length y should be same')
        sum = 0
        for i in range (0, len(x)) :
            sum += (x[i] - y[i]) ** 2
        
        return math.sqrt(sum)

    def __pairwiseDistance(self, X) :
        result = []
        for i in range(len(X)) :
            item = []
            for j in range(len(X)) :
                item.append(self.__euclidean(X.iloc[i], X.iloc[j]))
            result.append(item)
        return np.asarray(result)

    def fit(self, X) :
        # X is pandas.dataframe
        # raise error
        if not isinstance(X, pd.core.frame.DataFrame) :
            raise TypeError("X must be a pandas.core.frame.DataFrame")
        
        self.__X_train = X
        self.__distance = self.__pairwiseDistance(X)
        # print (self.__distance)
        
        temp_centroid = []

        for i in range(len(self.__distance)):
            temp_centroid.append(i)

        self.__clusters[0] = deepcopy(temp_centroid)

        for iterate in range(1, len(self.__distance)):
            min_distance = self.__distance[0][1]
            min_P1 = 0
            min_P2 = 1

            for i in range(len(self.__distance)):
                for j in range(i):
                    if min_distance > self.__distance[i][j]:
                        min_distance = self.__distance[i][j]
                        min_P2 = i
                        min_P1 = j

            if (self.__linkage == "single"):
                # do single linkage agglomerative here
                for i in range(len(self.__distance)):
                    if i != min_P1:
                        distance_change = min(self.__distance[min_P1][i], self.__distance[min_P2][i])
                        self.__distance[i][min_P1] = distance_change
                        self.__distance[min_P1][i] = distance_change

            elif (self.__linkage == "complete"):
                # do complete linkage agglomerative here
                for i in range(len(self.__distance)):
                    if i != min_P1:
                        distance_change = max(self.__distance[min_P1][i], self.__distance[min_P2][i])
                        self.__distance[i][min_P1] = distance_change
                        self.__distance[min_P1][i] = distance_change

            elif (self.__linkage == "average"):
                # do average linkage agglomerative here
                for i in range(len(self.__distance)):
                    if (i != min_P1 and i != min_P2):
                        distance_change = (self.__distance[min_P1][i] + self.__distance[min_P2][i])/2
                        self.__distance[i][min_P1] = distance_change
                        self.__distance[min_P1][i] = distance_change

            elif (self.__linkage == "average-group"):
                # do average-group linkage agglomerative here
                for i in range(len(self.__distance)):
                    if(i != min_P1 and i != min_P2):
                        if (i in self.__clusters[iterate-1]) and (min_P1 in self.__clusters[iterate-1]):
                            indices_0 = [o for o, u in enumerate(self.__clusters[iterate-1]) if u == i]
                            indices_1 = [o for o, u in enumerate(self.__clusters[iterate-1]) if u == min_P1]

                            sumDist = 0
                            for index1 in indices_1:
                                for index2 in indices_0:
                                    sumDist += self.__euclidean(self.__X_train.iloc[index1],self.__X_train.iloc[index2])
                            meanDist = sumDist / (len(indices_1) + len(indices_0))
                            self.__distance[min_P1][i] = meanDist
                            self.__distance[i][min_P1] = meanDist

            for i in range(len(self.__distance)):
                self.__distance[min_P2][i] = sys.maxsize
                self.__distance[i][min_P2] = sys.maxsize
                  
            for i in range(len(temp_centroid)):
                if(temp_centroid[i] == max(min_P1, min_P2)):
                    temp_centroid[i] = min(min_P1, min_P2)

            self.__clusters[iterate] = deepcopy(temp_centroid)
        # print (self.__clusters)

    def predict(self, X) :
        # X is pandas.dataframe
        # if not isinstance(X, pd.core.frame.DataFrame) :
        #     raise TypeError("X must be a pandas.core.frame.DataFrame")
        # elif X.select_dtypes(exclude=['number']).empty :
        #     raise TypeError("X must be all number")

        self.__X_train = X
        self.fit(self.__X_train)

        self.__iteration_number = self.__distance.shape[0] - self.__number_of_cluster
        return self.__clusters[self.__iteration_number]
