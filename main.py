# General Library
import pandas as pd
import numpy as np

# Clustering Implementation
from KMeans import KMeans
from dbscan import DBSCAN
from Metric import clustering_accuracy_score

# Sklearn Library
from sklearn import datasets

from sklearn.cluster import KMeans as sklearn_KMeans
from sklearn.cluster import AgglomerativeClustering as sklearn_AgglomerativeClustering
from sklearn.cluster import DBSCAN as sklearn_DBSCAN

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# Uncomment below code to load iris
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# read data
file_path = "iris.data"
df = pd.read_csv(file_path, header=None)

X = df.iloc[:,:-1]
y_temp = df.iloc[:,-1:]
y = y_temp.replace({'Iris-setosa' : 0, 'Iris-versicolor' : 1, 'Iris-virginica' : 2})

number_of_cluster = 3
kf = StratifiedKFold(n_splits=2)
kmeans = KMeans(number_of_cluster)
sk_kmeans = sklearn_KMeans(n_clusters=number_of_cluster)

k = 1
for train_index, test_index in kf.split(X, y):
    print (str(k) + '-fold')
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]
    kmeans.fit(np.asarray(X_train))
    result = kmeans.predict(np.asarray(X_test))
    accuracy, dict = clustering_accuracy_score(np.asarray(y_test), np.asarray(result))
    print ('Accuracy\t', accuracy)
    print ('Format : {Real class : cluster}')
    print ('Dict\t\t', str(dict))
    print ()
    sk_kmeans.fit(X_train)
    sk_result = sk_kmeans.predict(X_test)
    sk_accuracy, sk_dict = clustering_accuracy_score(np.asarray(y_test), np.asarray(sk_result))
    print ('Accuracy\t', sk_accuracy)
    print ('Format : {Real class : cluster}')
    print ('Dict\t\t', str(sk_dict))
    print ()
    k += 1