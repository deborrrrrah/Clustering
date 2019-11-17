# General Library
import pandas as pd
import numpy as np

# Clustering Implementation
from KMeans import KMeans
from dbscan import DBSCAN
from Agglomerative import Agglomerative
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

# KMeans model
kmeans = KMeans(number_of_cluster)
sk_kmeans = sklearn_KMeans(n_clusters=number_of_cluster)

# Agglomerative parameter for linkage
linkage_list = ['single', 'complete', 'average', 'average-group']

# DBSCAN model
epss = [0.5, 1]
min_ptss = [4, 5]

# Accuracy Mean Count Initialization
# Accuracy Mean Count Initialization
kmeans_accuracy = 0
sk_kmeans_accuracy = 0
agglo_accuracy_single = 0
agglo_accuracy_complete = 0
agglo_accuracy_average = 0
agglo_accuracy_average_group = 0
sk_agglo_accuracy_single = 0
sk_agglo_accuracy_complete = 0
sk_agglo_accuracy_average = 0
dbscan_accuracy = 0
sk_dbscan_accuracy = 0

print ('=== ACCURACY FROM PREDICT ===')
print ()

k = 0
for train_index, test_index in kf.split(X, y):
    print (str(k) + '-fold')
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]

    # KMeans 
    kmeans.fit(np.asarray(X_train))
    result = kmeans.predict(np.asarray(X_test))
    accuracy, dict = clustering_accuracy_score(np.asarray(y_test), np.asarray(result))
    kmeans_accuracy += accuracy
    print ('KMeans')
    print ('Accuracy\t', accuracy)
    print ('Format {Real class : cluster}')
    print ('Dict\t\t', str(dict))
    print ()

    sk_kmeans.fit(X_train)
    result = sk_kmeans.predict(X_test)
    accuracy, dict = clustering_accuracy_score(np.asarray(y_test), np.asarray(result))
    sk_kmeans_accuracy += accuracy
    print ('Sklearn KMeans')
    print ('Accuracy\t', accuracy)
    print ('Format {Real class : cluster}')
    print ('Dict\t\t', str(dict))
    print ()

    # Agglomerative
    for linkage_type in linkage_list :
        agglo = Agglomerative(number_of_cluster, linkage_type)
        agglo.fit(X_train)
        result = agglo.predict(X_test)
        accuracy, dict = clustering_accuracy_score(np.asarray(y_test), np.asarray(result))
        if linkage_type == 'single' :
            agglo_accuracy_single += accuracy
        elif linkage_type == 'complete' :
            agglo_accuracy_complete += accuracy
        elif linkage_type == 'average' :
            agglo_accuracy_average += accuracy
        elif linkage_type == 'average-group' :
            agglo_accuracy_average_group += accuracy
        print ('Agglomerative - ' + str(linkage_type))
        print ('Accuracy\t', accuracy)
        print ('Format {Real class : cluster}')
        print ('Dict\t\t', str(dict))
        print ()
    
    # DBSCAN
    for i in range (0, len(epss)) :
        eps = epss[i]
        min_pts = min_ptss[i]

        dbscan = DBSCAN(eps, min_pts)
        sk_dbscan = sklearn_DBSCAN(eps=eps, min_samples=min_pts)

        dbscan.fit(X_train)
        result = dbscan.predict(X_test)
        accuracy, dict = clustering_accuracy_score(np.asarray(y_test), np.asarray(result))
        dbscan_accuracy += accuracy
        print ('DBSCAN')
        print ('Epsilon : {} Min Points : {}'.format(eps, min_pts))
        print ('Accuracy\t', accuracy)
        print ('Format {Real class : cluster}')
        print ('Dict\t\t', str(dict))
        print ()

    k += 1

print ('RESULT')
print ('KMeans\t\t\t', kmeans_accuracy / k)
print ('SKlearns KMeans\t\t', sk_kmeans_accuracy / k)
print ('Agglomerative Single\t\t', agglo_accuracy_single / k)
print ('Agglomerative Complete\t\t', agglo_accuracy_complete / k)
print ('Agglomerative Average\t\t', agglo_accuracy_average / k)
print ('Agglomerative Average Group\t', agglo_accuracy_average_group / k)
print ('DBSCAN\t\t\t', dbscan_accuracy / (k * len(epss)))

print ()
print ('=== COMPARE TO SKLEARN ===')
print ()

kmeans.fit(np.asarray(X))
result = kmeans.predict(np.asarray(X))
accuracy, dict = clustering_accuracy_score(np.asarray(y), np.asarray(result))
print ('KMeans')
print ('Accuracy\t', accuracy)
print ('Format {Real class : cluster}')
print ('Dict\t\t', str(dict))
print ()

result = sk_kmeans.fit_predict(X)
accuracy, dict = clustering_accuracy_score(np.asarray(y), np.asarray(result))
print ('Sklearn KMeans')
print ('Accuracy\t', accuracy)
print ('Format {Real class : cluster}')
print ('Dict\t\t', str(dict))
print ()

# Agglomerative
for linkage_type in linkage_list :
    agglo = Agglomerative(number_of_cluster, linkage_type)
    result = agglo.predict(X)
    accuracy, dict = clustering_accuracy_score(np.asarray(y), np.asarray(result))
    print ('Agglomerative - ' + str(linkage_type))
    print ('Accuracy\t', accuracy)
    print ('Format {Real class : cluster}')
    print ('Dict\t\t', str(dict))
    print ()
    if linkage_type != 'average-group' :
        sk_agglo = sklearn_AgglomerativeClustering(n_clusters=number_of_cluster, linkage=linkage_type)
        result = sk_agglo.fit_predict(X)
        accuracy, dict = clustering_accuracy_score(np.asarray(y), np.asarray(result))
        print ('Sklearn Agglomerative - ' + str(linkage_type))
        print ('Accuracy\t', accuracy)
        print ('Format {Real class : cluster}')
        print ('Dict\t\t', str(dict))
        print ()

for i in range (0, len(epss)) :
    eps = epss[i]
    min_pts = min_ptss[i]

    dbscan = DBSCAN(eps, min_pts)
    sk_dbscan = sklearn_DBSCAN(eps=eps, min_samples=min_pts)

    dbscan.fit(X)
    result = dbscan.labels
    accuracy, dict = clustering_accuracy_score(np.asarray(y), np.asarray(result))
    print ('DBSCAN')
    print ('Epsilon : {} Min Points : {}'.format(eps, min_pts))
    print ('Accuracy\t', accuracy)
    print ('Format {Real class : cluster}')
    print ('Dict\t\t', str(dict))
    print ()

    labels_sklearn = sk_dbscan.fit_predict(X)
    accuracy, dict = clustering_accuracy_score(np.asarray(y), np.asarray(result))
    print ('Sklearn DBSCAN')
    print ('Epsilon : {} Min Points : {}'.format(eps, min_pts))
    print ('Accuracy\t', accuracy)
    print ('Format {Real class : cluster}')
    print ('Dict\t\t', str(dict))
    print ()
