import numpy as np

class KMeans:
    def __init__(self, k, init='random'):
        self.k = k
        self.init = init
        self.centroids = None

    def initialize_centroids(self, data):
        if self.init == 'random':
            indices = np.random.choice(len(data), self.k, replace=False)
            self.centroids = data[indices]
        elif self.init == 'farthest_first':
            pass  # Implement this
        elif self.init == 'kmeans++':
            pass  # Implement this

    def assign_clusters(self, data):
        distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def update_centroids(self, data, clusters):
        new_centroids = []
        for i in range(self.k):
            cluster_data = data[clusters == i]
            if len(cluster_data) > 0:
                new_centroids.append(np.mean(cluster_data, axis=0))
        self.centroids = np.array(new_centroids)

    def fit(self, data):
        self.initialize_centroids(data)
        for _ in range(100):
            clusters = self.assign_clusters(data)
            old_centroids = self.centroids.copy()
            self.update_centroids(data, clusters)
            if np.all(old_centroids == self.centroids):
                break
        return self.centroids, clusters
