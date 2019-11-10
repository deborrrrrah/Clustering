# General Library
import pandas as pd
import numpy as np

# Clustering Implementation
from KMeans import KMeans
from dbscan import DBSCAN

# Sklearn Library
from sklearn import datasets

from sklearn.cluster import KMeans as sklearn_KMeans
from sklearn.cluster import AgglomerativeClustering as sklearn_AgglomerativeClustering
from sklearn.cluster import DBSCAN as sklearn_DBSCAN

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# Uncomment below code to load iris
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# Comment below
X = [0, 1]
y = [0]

kf = KFold(n_splits=2)
kmeans = KMeans(3)
for train_index, test_index in kf.split(X):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    kmeans.fit(X_train)
    result = kmeans.predict(X_test)