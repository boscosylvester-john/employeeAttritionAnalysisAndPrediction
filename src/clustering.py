import collections
import pandas as pd
import math as m
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

data_file = "./data/preprocessed_data.csv"

def preprocess(data):
    data.drop(data.loc[data['Attrition'] == 1].index, inplace=True)
    data = data.reset_index(drop=True)
    data = data.drop(columns=['EmployeeID', 'Attrition'])
    return data


def generate_clusters():
    data = pd.read_csv(data_file)
    data = preprocess(data)
    num_clusters = find_optimal_number_of_clusters(data)
    centroids = form_clusters(data, num_clusters)
    return centroids


def find_optimal_number_of_clusters(data):
    sse = []
    for k in range(2, 21, 2):
        kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
        kmeans.fit(data)
        sse.append(kmeans.inertia_)

    plt.plot(range(2, 21, 2), sse)
    plt.xticks(range(2, 21, 2))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()

    kl = KneeLocator(range(2, 21, 2), sse, curve="convex", direction="decreasing")
    print("Optimal number of clusters: ", kl.elbow)
    return kl.elbow


def form_clusters(data, num_clusters):
    model = KMeans(n_clusters=num_clusters, random_state=42).fit(data)
    centroids = model.cluster_centers_
    assignments = model.labels_

    # print(centroids, len(centroids), len(centroids[0]))
    # print(assignments, len(assignments))

    data['Cluster'] = assignments
    pivot = data.pivot_table(index='Cluster', aggfunc=np.mean)
    counters = collections.defaultdict(int)
    percentages = []
    for i in range(0, data['Cluster'].size):
        counters[data['Cluster'][i]] += 1

    # print(counters)
    for i in range(0, len(counters)):
        percentages.append(counters[i] / data['Cluster'].size)
        for j in range(len(percentages)):
            pivot.loc[j, '% of Attrition'] = percentages[j] * 100

    assert pivot['% of Attrition'].sum() == 100, 'Sum of percentage attrition should be 1'

    print(pivot.sort_values(by='% of Attrition', axis=0).T)

    return centroids
