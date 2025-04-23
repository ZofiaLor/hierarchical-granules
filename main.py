import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from fuzzy_cmeans import FuzzyCMeans as FCM
import seaborn as sns
import random

# load the data and convert it to an appropriate format
iris = load_iris()
x = iris.data[:, 2].tolist()
y = iris.data[:, 3].tolist()
data = np.array(list(zip(x, y)))

# perform hierarchical clustering and display the results
cluster = AgglomerativeClustering(n_clusters=2)
labels = cluster.fit_predict(data)
plt.scatter(x, y, c=labels)
plt.show()

# display a dendrogram visualizing the clustering
dendro = linkage(data, method='ward', metric='euclidean')
dendrogram(dendro)
plt.show()

# perform fuzzy c-means clustering and display the results
fcm = FCM(n_clusters=10)
fcm.fit(data)
labels_clusters = fcm.predict(data)
plt.scatter(x, y, c=labels_clusters)
plt.scatter(fcm.cluster_centers_[:, 0], fcm.cluster_centers_[:, 1], c='red')
plt.show()

