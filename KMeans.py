import pandas as pd
import numpy as np
from math import sqrt
from copy import deepcopy
import array, collections

IntegerTypes = (int)
NumberTypes = ['int32', 'int64', 'float32', 'float64']
ArrayTypes = (np.ndarray)

class KMeans:
    __centroid = []

    def __init__(self, number_of_cluster) :
        if not isinstance(number_of_cluster, IntegerTypes) :
            raise TypeError('number_of_cluster must be a integer')
        self.__number_of_cluster = number_of_cluster
    
    def __countEuclideanDistance(self, x, centroid) :
        diff = np.subtract(x, centroid)
        sum_diff = 0
        for dif in diff :
            sum_diff += (dif * dif)
        return sqrt(sum_diff)
    
    def __addArray(self, arr1, arr2) :
        result = []
        for i in range(len(arr1)) :
            result.append(arr1[i] + arr2[i])
        return np.asarray(result)

    def __clusterObject(self) :
        # Put object to one cluster
        cluster_result = np.full(self.__X_train.shape[0], -1)
        for i in range (self.__X_train.shape[0]):
            minimum_centroid = self.__centroid[0]
            minimum_index = 0
            for j in range (self.__centroid.shape[0] - 1):
                if (self.__countEuclideanDistance(self.__X_train[i], self.__centroid[j+1]) < self.__countEuclideanDistance(self.__X_train[i], minimum_centroid)) :
                    minimum_centroid = self.__centroid[j+1]
                    minimum_index = j+1
            cluster_result[i] = minimum_index
        return cluster_result

    def fit(self, X) :
        if not isinstance(X, ArrayTypes) :
            raise TypeError("X must be a numpy ndarray")
        elif isinstance(X, ArrayTypes):
            for i in range(len(X)):
                if X[i].dtype not in NumberTypes:
                    raise TypeError("Attribute must be numerical")
        self.__X_train = X

        # Result initialization
        self.__centroid = np.full((self.__number_of_cluster, self.__X_train.shape[1]), -1)

        # Choose random centroid
        for i in range (self.__number_of_cluster) :
            centroid_idx = np.random.randint(self.__X_train.shape[0], size=1)[0]
            self.__centroid[i] = self.__X_train[centroid_idx]
        self.__centroid = np.asarray(self.__centroid)
        
        prev_cluster = np.full(self.__X_train.shape[0], -999)
        cluster_result = np.full(self.__X_train.shape[0], -99)

        # Cluster objects to nearest cluster
        while (1) :
            prev_cluster = deepcopy(cluster_result)
            
            cluster_result = self.__clusterObject()
            
            all_cluster_list = []
            
            # Calculate new centroid
            for i in range (self.__number_of_cluster) :
                cluster_result_modified = np.full(self.__X_train.shape[1], 0)
                result = np.where(cluster_result == i)
                count = 0
                if (len(result[0]) > 0) :
                    for idx in result[0] :
                        cluster_result_modified = self.__addArray(cluster_result_modified, self.__X_train[idx])
                        count += 1
                    mean_result = np.divide(np.asarray(cluster_result_modified), count)
                    all_cluster_list.append(mean_result)
                else :
                    all_cluster_list.append(cluster_result_modified)

            all_cluster_result = np.asarray(all_cluster_list)

            # Update centroid
            for i in range(len(self.__centroid)) :
                self.__centroid[i] = all_cluster_result[i]

            # End condition when clustering does not change
            if (np.all(prev_cluster == cluster_result)) :
                break
    
    def predict(self, X) :
        if not isinstance(X, ArrayTypes) :
            raise TypeError("X must be a numpy ndarray")
        elif isinstance(X, ArrayTypes):
            for i in range(len(X)):
                if X[i].dtype not in NumberTypes:
                    raise TypeError("Attribute must be numerical")
        self.__X_train = X
        result = self.__clusterObject()
        return result