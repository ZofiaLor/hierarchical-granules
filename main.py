import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from fuzzy_cmeans import FuzzyCMeans as FCM
import time
import math
import os


class DataEntry(object):
    def __init__(self, data, name, dim=2):
        self.x = []
        self.y = []
        self.z = []
        data = data.splitlines()
        for line in data:
            coords = line.split()
            self.x.append(float(coords[0]))
            self.y.append(float(coords[1]))
            if dim > 2:
                self.z.append(float(coords[2]))
        if dim > 2:
            self.data = np.array(list(zip(self.x, self.y, self.z)))
        else:
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
        self.variances = []

    def cluster_data(self, repeat: int):
        self.hierarchy_times = []
        for i in range(repeat):
            cluster = AgglomerativeClustering(n_clusters=self.clusters_number, linkage='single')
            start = time.time()
            self.labels = cluster.fit_predict(self.data)
            end = time.time()
            self.hierarchy_times.append((end - start) * 1000)

    def compute_variances(self, dim=2):
        if self.fcm is None:
            return
        self.variances = np.empty(shape=(self.granules_number, dim))
        for c in range(self.granules_number):
            for a in range(dim):
                numerator = sum(
                    (self.fcm.U_[i, c] ** 2) * ((self.fcm.X_[i, a] - self.fcm.cluster_centers_[c, a]) ** 2) for i in
                    range(self.length))
                denominator = sum(self.fcm.U_[i, c] ** 2 for i in range(self.length))
                self.variances[c][a] = np.sqrt(numerator / denominator)

    def plot_ellipses(self, dim=2):
        if self.fcm is None:
            return
        colors = ['slateblue', 'limegreen', 'dodgerblue', 'mediumorchid']
        if dim == 2:
            t = np.linspace(0, 2 * np.pi)
            plt.scatter(self.x, self.y, c='lightgray')
            for i in range(self.granules_number):
                plt.plot(self.fcm.cluster_centers_[i, 0] + 2 * self.variances[i][0] * np.cos(t),
                         self.fcm.cluster_centers_[i, 1] + 2 * self.variances[i][1] * np.sin(t),
                         color=colors[self.labels_fcm[i]])
            plt.scatter(self.fcm.cluster_centers_[:, 0], self.fcm.cluster_centers_[:, 1], c='deeppink')
            plt.savefig("img/" + self.name + "_var.pdf")
        elif dim == 3:
            t = np.linspace(0, 2 * np.pi, num=10)
            p = np.linspace(0, np.pi, num=10)
            t, p = np.meshgrid(t, p)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter3D(self.x, self.y, self.z, c='lightgray')
            for i in range(self.granules_number):
                sx = self.fcm.cluster_centers_[i, 0] + 2 * self.variances[i][0] * np.sin(p) * np.cos(t)
                sy = self.fcm.cluster_centers_[i, 1] + 2 * self.variances[i][1] * np.sin(p) * np.sin(t)
                sz = self.fcm.cluster_centers_[i, 2] + 2 * self.variances[i][2] * np.cos(p)
                ax.plot_surface(sx, sy, sz, color=colors[self.labels_fcm[i]], alpha=0.5)
            ax.scatter3D(self.fcm.cluster_centers_[:, 0], self.fcm.cluster_centers_[:, 1], self.fcm.cluster_centers_[:, 2], c='deeppink')
            fig.savefig("img/" + self.name + "_var.pdf")

    def fcm_data(self, repeat: int):
        self.labels_clusters = []
        for i in range(repeat):
            self.fcm = FCM(n_clusters=self.granules_number, max_iter=20)
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
            if "spheres" in file.name:
                fullData[file.name[:-5]] = DataEntry(f.read(), file.name[:-5], 3)
            else:
                fullData[file.name[:-5]] = DataEntry(f.read(), file.name[:-5])
            for key, value in num_of_clusters.items():
                if key in file.name:
                    fullData[file.name[:-5]].clusters_number = value
                    break

user_input = ""


def plot_results(name, dim=2):
    if dim == 2:
        plt.figure()
        plt.scatter(fullData[user_input].x, fullData[user_input].y)
        plt.title("Original data")
        plt.savefig("img/" + name + "_original.pdf")
        plt.figure()
        plt.scatter(fullData[user_input].x, fullData[user_input].y, c=fullData[user_input].labels)
        plt.title("Hierarchical clustering of non-granulated data")
        plt.savefig("img/" + name + "_hng.pdf")
        plt.figure()
        plt.scatter(fullData[user_input].x, fullData[user_input].y, c=fullData[user_input].labels_clusters)
        plt.scatter(fullData[user_input].fcm.cluster_centers_[:, 0],
                           fullData[user_input].fcm.cluster_centers_[:, 1],
                           c='deeppink')
        plt.title("FCM clusters")
        plt.savefig("img/" + name + "_fcm.pdf")
        plt.figure()
        plt.scatter(fullData[user_input].fcm.cluster_centers_[:, 0],
                           fullData[user_input].fcm.cluster_centers_[:, 1],
                           c=fullData[user_input].labels_fcm)
        plt.title("Hierarchical clustering of granulated data")
        plt.savefig("img/" + name + "_hg.pdf")
    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter3D(fullData[user_input].x, fullData[user_input].y, fullData[user_input].z)
        ax.set_title('Original data')
        fig.savefig("img/" + name + "_original.pdf")
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter3D(fullData[user_input].x, fullData[user_input].y, fullData[user_input].z,
                     c=fullData[user_input].labels)
        ax.set_title('Hierarchical clustering of non-granulated data')
        fig.savefig("img/" + name + "_hng.pdf")
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter3D(fullData[user_input].x, fullData[user_input].y, fullData[user_input].z,
                     c=fullData[user_input].labels_clusters)
        ax.scatter3D(fullData[user_input].fcm.cluster_centers_[:, 0], fullData[user_input].fcm.cluster_centers_[:, 1],
                     fullData[user_input].fcm.cluster_centers_[:, 2], c='deeppink')
        ax.set_title('FCM clusters')
        fig.savefig("img/" + name + "_fcm.pdf")
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter3D(fullData[user_input].fcm.cluster_centers_[:, 0],
                     fullData[user_input].fcm.cluster_centers_[:, 1],
                     fullData[user_input].fcm.cluster_centers_[:, 2],
                     c=fullData[user_input].labels_fcm)
        ax.set_title('Hierarchical clustering of granulated data')
        fig.savefig("img/" + name + "_hg.pdf")


def plot_variance(dim=2):
    fullData[user_input].compute_variances(dim)
    fullData[user_input].plot_ellipses(dim)


while user_input != "exit":
    user_input = input("Enter the name of the data file for which to perform the clustering (eg. 'blobs1000') or type "
                       "'exit' to finish the program ")
    if user_input in names:
        fullData[user_input].cluster_and_measure(1)
        print("Average time of hierarchical clustering of non-granulated data: ",
              np.mean(fullData[user_input].hierarchy_times), "ms")
        print("Average time of fuzzy c-means clustering of data: ",
              np.mean(fullData[user_input].fcm_times), "ms")
        print("Average time of hierarchical clustering of granulated data: ",
              np.mean(fullData[user_input].fcm_hierarchy_times), "ms")
        if "spheres" in user_input:
            plot_results(user_input, 3)
            plot_variance(3)
        else:
            plot_results(user_input, 2)
            plot_variance(2)
    else:
        print("A file by that name does not exist")
