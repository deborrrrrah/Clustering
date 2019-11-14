import math
import pandas as pd
import operator

class DBSCAN :
    labels = [] # -1 for noise
    labels_predict = []
    __X = []
    __unvisited = set()
    __neighbors = []
    __neighbors_predict = []
    __curr_cluster = -1

    def __init__(self, eps, min_pts) :
        self.__eps = eps
        self.__min_pts = min_pts

    # helper function
    @staticmethod
    def __eucledian(x, y) :
    # input  : x, y : points (row of the dataset)   {pandas.Series}
    # output : the distance of two points           {float}
        if len(x) != len(y) :
            raise Exception('length x and length y should be same')
        sum = 0
        for i in range (0, len(x)) :
            sum += (x[i] - y[i]) ** 2
        
        return math.sqrt(sum)

    def __init_neighbors(self) :
    # find list of neighbors' indices
        for i in range (0, len(self.__X.index)) :
            for j in range (0, i) :
                if i == j :
                    pass
                else :
                    distance = self.__eucledian(self.__X.loc[i], self.__X.loc[j])
                    if distance <= self.__eps :
                        self.__neighbors[i].append(j)
                        self.__neighbors[j].append(i)    

    def __expand_cluster(self, p, pts) :
    # expand cluster until no point added
    # input :   p : index of current core object        {int}
    #           pts : list of neighbors' indices of p   {int}
        self.labels[p] = self.__curr_cluster
        for pt in pts :
            if pt not in self.__unvisited :
                self.__unvisited.discard(pt)
                nei_pts = self.__neighbors[pt]
                if len(nei_pts) >= self.__min_pts :
                    self.__expand_cluster(pt, nei_pts)
            if self.labels[pt] == -1 :
                self.labels[pt] = self.__curr_cluster

    # end of helper function

    # fit function
    def fit(self, X) :
    # input : X : dataset   {pandas.DataFrame || [[int]]}

        # check input
        if isinstance(X, pd.DataFrame) :
            self.__X = X
        elif isinstance(X, list) :
            self.__X = pd.DataFrame(data=X)
        else :
            raise Exception('X should be a pandas.Dataframe or list of list')
        
        len_X = len(self.__X.index)
        self.__unvisited = set(i for i in range(0, len_X))
        self.labels = [-1] * len_X
        self.__neighbors = [[i] for i in range(len_X)]
        
        self.__init_neighbors()

        while self.__unvisited :
            curr = self.__unvisited.pop()
            nei_pts = self.__neighbors[curr]
            if len(nei_pts) >= self.__min_pts :
                self.__curr_cluster += 1
                self.__expand_cluster(curr, nei_pts)

    def predict(self, X) :
    # method using nearest neighbor
    # input : X : dataset       {pandas.DataFrame || [[int]]}
    # output : list of labels   {[int]}

        # check input
        if isinstance(X, pd.DataFrame) :
            self.__X_predict = X
        elif isinstance(X, list) :
            self.__X_predict = pd.DataFrame(data=X)
        else :
            raise Exception('X should be a pandas.Dataframe or list of list')

        self.labels_predict = [-1] * len(self.__X_predict.index)

        for i in range (0, len(self.__X_predict.index)) :
            min_distance = -1
            min_label = -1
            for j in range (0, len(self.__X.index)) :
                distance = self.__eucledian(self.__X_predict.loc[i], self.__X.loc[j])
                if min_distance == -1 :
                    min_distance = distance
                    min_label = self.labels[j]
                elif distance < min_distance :
                    min_distance = distance
                    min_label = self.labels[j]
            self.labels_predict[i] = min_label

        return self.labels_predict