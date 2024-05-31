import numpy as np



class SumOfSquaredErrors():
    def __init__(self, dim, n_clusters, min=0.0, max=1.0):
        self.dim = dim
        self.min = min
        self.max = max
        self.n_clusters = n_clusters
        self.centroids = {}

    def sample(self):
        return np.random.uniform(low=self.min, high=self.max, size=self.dim)

    def evaluate(self, x, data):
        centroids = x.reshape(self.n_clusters, int(self.dim/self.n_clusters))
        self.centroids = dict(enumerate(centroids))
        centroid_distances = np.linalg.norm(data[:, None] - centroids, axis=2)
        clusters = {key: [] for key in self.centroids.keys()}
        clusters.fromkeys(clusters, [])
        for i in range(data.shape[0]):
            closest_centroid_idx = np.argmin(centroid_distances[i])
            clusters[closest_centroid_idx].append(data[i].tolist())

        sum_of_squared_errors = 0.0
        for idx in self.centroids:
            cluster_data = np.array(clusters[idx])
            if len(cluster_data) > 0:
                squared_distances = np.sum(np.power(cluster_data - self.centroids[idx], 2), axis=1)
                sum_of_squared_errors += np.sum(squared_distances)

        return sum_of_squared_errors

