from sklearn.metrics import accuracy_score
from copy import deepcopy
import numpy as np
from itertools import repeat, permutations

NumberTypes = ['int32', 'int64', 'float32', 'float64']
ArrayTypes = (np.ndarray)
dummy_number = -1

def clustering_accuracy_score(y_true, y_pred) :
    if not isinstance(y_true, ArrayTypes) :
        raise TypeError("y_true must be a numpy ndarray")
    elif isinstance(y_true, ArrayTypes):
        for i in range(len(y_true)):
            if y_true[i].dtype not in NumberTypes:
                raise TypeError("Class in y_true must be numerical")
    if not isinstance(y_pred, ArrayTypes) :
        raise TypeError("y_pred must be a numpy ndarray")
    elif isinstance(y_pred, ArrayTypes):
        for i in range(len(y_pred)):
            if y_pred[i].dtype not in NumberTypes:
                raise TypeError("Class in y_pred must be numerical")
    
    cluster_true = np.unique(y_true)
    cluster_pred = np.unique(y_pred)

    # delete -1
    cluster_true = cluster_true[cluster_true != -1]
    cluster_pred = cluster_pred[cluster_pred != -1]

    max_dict_result = dict(zip(cluster_true, repeat(None)))
    max_accuracy = dummy_number
    if (len(cluster_true) > len(cluster_pred)) :
        permutations_list = [dict(zip(cluster_true, x)) for x in permutations(cluster_pred,len(cluster_pred))]
    else :
        permutations_list = [dict(zip(cluster_true, x)) for x in permutations(cluster_pred,len(cluster_true))]
    for permutation in permutations_list :
        y_pred_temp = deepcopy(y_pred)
        flipped_permutation = dict(zip(permutation.values(), permutation.keys()))
        y_pred_temp = [flipped_permutation[value] if value != dummy_number else value for value in y_pred_temp]
        if (accuracy_score(y_true, y_pred_temp) > max_accuracy):
            max_accuracy = accuracy_score(y_true, y_pred_temp)
            max_dict_result = permutation
    return max_accuracy, max_dict_result