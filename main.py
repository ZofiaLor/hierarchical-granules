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

        self.hierarchy_times = []
        self.fcm_times = []
        self.fcm_hierarchy_times = []


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

plt.scatter(fullData["blobs1000"].x, fullData["blobs1000"].y)
plt.show()
# load the data and convert it to an appropriate format
iris = load_iris()
x = iris.data[:, 2].tolist()
y = iris.data[:, 3].tolist()
data = np.array(list(zip(x, y)))

# perform hierarchical clustering and display the results
cluster = AgglomerativeClustering(n_clusters=2, linkage='single')
start = time.time()
labels = cluster.fit_predict(data)
end = time.time()
print("Time of hierarchical clustering of non-granulated data: ", (end - start) * 1000, "ms")
# plt.scatter(x, y, c=labels)
# plt.show()

# display a dendrogram visualizing the clustering
dendro = linkage(data, method='single', metric='euclidean')
# dendrogram(dendro)
# plt.show()

# perform fuzzy c-means clustering and display the results
fcm = FCM(n_clusters=10)
start = time.time()
fcm.fit(data)
labels_clusters = fcm.predict(data)
end = time.time()
print("Time of fuzzy c-means clustering of data: ", (end - start) * 1000, "ms")
# plt.scatter(x, y, c=labels_clusters)
# plt.scatter(fcm.cluster_centers_[:, 0], fcm.cluster_centers_[:, 1], c='red')
# plt.show()

# hierarchical clustering of the granules (centers)
clusterFcm = AgglomerativeClustering(n_clusters=2, linkage='single')
start = time.time()
labelsFcm = cluster.fit_predict(fcm.cluster_centers_)
end = time.time()
print("Time of hierarchical clustering of granulated data: ", (end - start) * 1000, "ms")
# plt.scatter(fcm.cluster_centers_[:, 0], fcm.cluster_centers_[:, 1], c=labelsFcm)
# plt.show()

# display a dendrogram visualizing the clustering of granules
dendroFcm = linkage(fcm.cluster_centers_, method='single', metric='euclidean')
# dendrogram(dendroFcm)
# plt.show()

figure, axis = plt.subplots(2, 3)
axis[0, 0].scatter(x, y)
axis[0, 0].set_title("Original data")
axis[0, 1].scatter(x, y, c=labels)
axis[0, 1].set_title("Hierarchical clustering of non-granulated data")
dendrogram(dendro, ax=axis[0, 2])
axis[0, 2].set_title("Dendrogram for non-granulated data")
axis[1, 0].scatter(x, y, c=labels_clusters)
axis[1, 0].scatter(fcm.cluster_centers_[:, 0], fcm.cluster_centers_[:, 1], c='red')
axis[1, 0].set_title("FCM clusters")
axis[1, 1].scatter(fcm.cluster_centers_[:, 0], fcm.cluster_centers_[:, 1], c=labelsFcm)
axis[1, 1].set_title("Hierarchical clustering of granulated data")
dendrogram(dendroFcm, ax=axis[1, 2])
axis[1, 2].set_title("Dendrogram for granulated data")
plt.show()
