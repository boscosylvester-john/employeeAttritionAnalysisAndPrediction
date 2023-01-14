import pandas as pd
import numpy as np 
import math as m
import matplotlib.pyplot as plt
import utils
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def KNN_distance(data, input_id, threshold=m.inf):
    knn = NearestNeighbors(n_neighbors = 11, p = 2)
    data1 = data[list(set(data.columns).difference(set(['EmployeeID','Attrition'])))]
    knn.fit(data1)
    neighbours = knn.kneighbors(data1.loc[(data['EmployeeID'] == input_id)])
    output = {}
    for i in range(len(list(neighbours[1][0]))):
        if round(list(neighbours[0][0])[i],5)<threshold and data['EmployeeID'][list(neighbours[1][0])[i]]!=input_id:
            output[data['EmployeeID'][list(neighbours[1][0])[i]]] = round(list(neighbours[0][0])[i],5)
    # print(output)
    # plt.plot(neighbours[0][0])
    # plt.show()
  
    return output

#USAGE EXAMPLE
#KNN_distance(utils.get_preprocessed_dataset(), 1469740, threshold = 4.74)
