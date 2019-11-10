import pandas as pd
import numpy as np
from math import sqrt
from copy import deepcopy

IntegerTypes = (int)
ArrayTypes = (pd.core.series.Series, np.ndarray)

class Metric:

    def __init__(self, cluster_result, number_of_cluster) :
        self.__cluster_result = []
        for i in range (number_of_cluster) :
            number_of_objects_one_cluster = []
            text = 'Cluster ' + str(i + 1) + ' -> '
            for j in range (number_of_cluster) :
                indexes = np.where(cluster_result == i)[0]
                digit_indexes = np.where((indexes >= 50 * j) & (indexes < 50 * (j+1)))[0]
                text += 'class ' + str(j) + ': ' + str(len(digit_indexes))
                number_of_objects_one_cluster.append(len(digit_indexes))
                if (j < number_of_cluster-1) :
                    text += ', '
            self.__cluster_result.append(number_of_objects_one_cluster)
            print (text)
    
    def __recall_score(self) :
        for i in range(len(self.__cluster_result)) :
            cluster_index = int(np.where(self.__cluster_result[i] == np.max(self.__cluster_result[i]))[0])
            print ('Cluster ' + str(i) + ' with majority ' + str(cluster_index))
            true_positive = self.__cluster_result[i][cluster_index]
        actual_positive = 0
        for j in range(len(self.__cluster_result)) :
            actual_positive += self.__cluster_result[j][cluster_index]
        print ('Recall\t\t: ', true_positive/actual_positive)
        print ()
    
    def __specificity_score(self) :
        for i in range(len(self.__cluster_result)) :
            cluster_index = int(np.where(self.__cluster_result[i] == np.max(self.__cluster_result[i]))[0])
            print ('Cluster ' + str(i) + ' with majority ' + str(cluster_index))
            false_positive = np.sum(self.__cluster_result[i]) - self.__cluster_result[i][cluster_index]
        actual_negative = 0
        for j in range(len(self.__cluster_result)) :
            actual_positive += self.__cluster_result[j][cluster_index]
            actual_negative += np.sum(self.__cluster_result[j]) - self.__cluster_result[j][cluster_index]
        true_negative = actual_negative - false_positive
        print ('Specificity\t: ', true_negative/actual_negative)
        print ()

    def __accuracy_score(self) :
        for i in range(len(self.__cluster_result)) :
            cluster_index = int(np.where(self.__cluster_result[i] == np.max(self.__cluster_result[i]))[0])
            print ('Cluster ' + str(i) + ' with majority ' + str(cluster_index))
            true_positive = self.__cluster_result[i][cluster_index]
            false_positive = np.sum(self.__cluster_result[i]) - self.__cluster_result[i][cluster_index]
        actual_positive = 0
        actual_negative = 0
        for j in range(len(self.__cluster_result)) :
            actual_positive += self.__cluster_result[j][cluster_index]
            actual_negative += np.sum(self.__cluster_result[j]) - self.__cluster_result[j][cluster_index]
        true_negative = actual_negative - false_positive
        print ('Accuracy\t\t: ', (true_positive + true_negative)/(actual_positive + actual_negative))
        print ()