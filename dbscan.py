import math
import pandas as pd

class DBSCAN :
    self.labels = [] # -1 for noise
    self.__X = []
    self.__unvisited = set()
    self.__neighbors = []
    self.__curr_cluster = -1

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
                    distance = (__eucledian(self.__X.loc[i], self.__X.loc[j]))
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
            if labels[pt] == -1 :
                labels[pt] = self.__curr_cluster

    # end of helper function

    # fit function
    def fit(self, X) :
    # input : X : dataset {pandas.DataFrame}
        self.__X = X
        
        len_X = len(self.__X.index)
        self.__unvisited = set(i for i in range(0, len_X))
        self.__labels = [-1] * len_X
        self.__neighbors = [[i] for i in range(len_X)]
        
        self.__init_neighbors()

        while self.__unvisited :
            curr = self.__unvisited.pop()
            nei_pts = self.__neighbors[i]
            if len(nei_pts) >= self.__min_pts :
                self.__curr_cluster += 1
                self.__expand_cluster(i, nei_pts)