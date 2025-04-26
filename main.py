import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from fuzzy_cmeans import FuzzyCMeans as FCM
import time
import seaborn as sns
import random

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
print("Time of hierarchical clustering of non-granulated data: ", (end - start) * 0.001, "ms")
plt.scatter(x, y, c=labels)
plt.show()

# display a dendrogram visualizing the clustering
dendro = linkage(data, method='single', metric='euclidean')
dendrogram(dendro)
plt.show()

# perform fuzzy c-means clustering and display the results
fcm = FCM(n_clusters=10)
start = time.time()
fcm.fit(data)
labels_clusters = fcm.predict(data)
end = time.time()
print("Time of fuzzy c-means clustering of data: ", (end - start) * 0.001, "ms")
plt.scatter(x, y, c=labels_clusters)
plt.scatter(fcm.cluster_centers_[:, 0], fcm.cluster_centers_[:, 1], c='red')
plt.show()

# hierarchical clustering of the granules (centers)
clusterFcm = AgglomerativeClustering(n_clusters=2, linkage='single')
start = time.time()
labelsFcm = cluster.fit_predict(fcm.cluster_centers_)
end = time.time()
print("Time of hierarchical clustering of granulated data: ", (end - start) * 0.001, "ms")
plt.scatter(fcm.cluster_centers_[:, 0], fcm.cluster_centers_[:, 1], c=labelsFcm)
plt.show()

# display a dendrogram visualizing the clustering of granules
dendroFcm = linkage(fcm.cluster_centers_, method='single', metric='euclidean')
dendrogram(dendroFcm)
plt.show()

