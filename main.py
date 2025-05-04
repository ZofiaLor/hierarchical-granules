import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from fuzzy_cmeans import FuzzyCMeans as FCM
import time
import seaborn as sns
import random
import os


class DataEntry(object):
    def __init__(self, data, name):
        self.x = []
        self.y = []
        data = data.splitlines()
        for line in data:
            coords = line.split()
            self.x.append(float(coords[0]))
            self.y.append(float(coords[1]))
        self.data = np.array(list(zip(self.x, self.y)))
        self.name = name
        self.length = len(self.x)
        self.clusters_number = 0
        self.granules_number = 100

        self.hierarchy_times = []
        self.fcm_times = []
        self.fcm_hierarchy_times = []

        self.labels = []
        self.labels_clusters = []
        self.labels_fcm = []

        self.fcm = None

    def cluster_data(self, repeat: int):
        self.hierarchy_times = []
        for i in range(repeat):
            cluster = AgglomerativeClustering(n_clusters=self.clusters_number, linkage='single')
            start = time.time()
            self.labels = cluster.fit_predict(self.data)
            end = time.time()
            self.hierarchy_times.append((end - start) * 1000)

    def fcm_data(self, repeat: int):
        self.labels_clusters = []
        for i in range(repeat):
            self.fcm = FCM(n_clusters=self.granules_number)
            start = time.time()
            self.fcm.fit(self.data)
            self.labels_clusters = self.fcm.predict(self.data)
            end = time.time()
            self.fcm_times.append((end - start) * 1000)

    def cluster_granules(self, repeat: int):
        if self.fcm is None:
            return
        self.labels_fcm = []
        for i in range(repeat):
            clusterFcm = AgglomerativeClustering(n_clusters=self.clusters_number, linkage='single')
            start = time.time()
            self.labels_fcm = clusterFcm.fit_predict(self.fcm.cluster_centers_)
            end = time.time()
            self.fcm_hierarchy_times.append((end - start) * 1000)

    def cluster_and_measure(self, repeat: int):
        self.cluster_data(repeat)
        self.fcm_data(repeat)
        self.cluster_granules(repeat)


fullData = {}
names = []

# Values picked by visually deciding the best number of clusters for a given shape
num_of_clusters = {"blobs": 3, "circles": 2, "corners": 4, "crescents": 2, "laguna": 3, "spheres": 2}

for folder in os.scandir("dane"):
    for file in os.scandir(folder.path):
        with open(file.path) as f:
            # Assumption: all folders contain only files with .data extension
            names.append(file.name[:-5])
            fullData[file.name[:-5]] = DataEntry(f.read(), file.name[:-5])
            for key, value in num_of_clusters.items():
                if key in file.name:
                    fullData[file.name[:-5]].clusters_number = value
                    break

user_input = ""
while user_input != "exit":
    user_input = input("Enter the name of the data file for which to perform the clustering (eg. 'blobs1000') or type "
                       "'exit' to finish the program ")
    if user_input in names:
        fullData[user_input].cluster_and_measure(10)
        print("Average time of hierarchical clustering of non-granulated data: ",
              np.mean(fullData[user_input].hierarchy_times), "ms")
        print("Average time of fuzzy c-means clustering of data: ",
              np.mean(fullData[user_input].fcm_times), "ms")
        print("Average time of hierarchical clustering of granulated data: ",
              np.mean(fullData[user_input].fcm_hierarchy_times), "ms")
        figure, axis = plt.subplots(2, 2)
        axis[0, 0].scatter(fullData[user_input].x, fullData[user_input].y)
        axis[0, 0].set_title("Original data")
        axis[0, 1].scatter(fullData[user_input].x, fullData[user_input].y, c=fullData[user_input].labels)
        axis[0, 1].set_title("Hierarchical clustering of non-granulated data")
        axis[1, 0].scatter(fullData[user_input].x, fullData[user_input].y, c=fullData[user_input].labels_clusters)
        axis[1, 0].scatter(fullData[user_input].fcm.cluster_centers_[:, 0], fullData[user_input].fcm.cluster_centers_[:, 1], c='red')
        axis[1, 0].set_title("FCM clusters")
        axis[1, 1].scatter(fullData[user_input].fcm.cluster_centers_[:, 0], fullData[user_input].fcm.cluster_centers_[:, 1], c=fullData[user_input].labels_fcm)
        axis[1, 1].set_title("Hierarchical clustering of granulated data")
        plt.show()
    else:
        print("A file by that name does not exist")
