# General Library
import pandas as pd
import numpy as np

# Clustering Implementation
from KMeans import KMeans
from dbscan import DBSCAN
from Metric import Metric

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

# read data
file_path = "iris.data"
df = pd.read_csv(file_path, header=None)

X = df.iloc[:,:-1]
y = df.iloc[:,-1:]

number_of_cluster = 3
kf = KFold(n_splits=2)
kmeans = KMeans(number_of_cluster)

for train_index, test_index in kf.split(X):
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]
    kmeans.fit(X_train)
    result = kmeans.predict(X_test)
    metric = Metric(result, number_of_cluster)