import numpy as np
import pandas as pd


def initialize_centroids_forgy(data, k):
    # Randomly initialize centroids by picking k data points
    indices = np.random.choice(data.shape[0], size=k, replace=False)
    return data[indices, :]


def initialize_centroids_kmeans_pp1(data, k):

    # Initialize centroids using k-means++ initialization
    centroids = []
    centroids.append(data[np.random.choice(data.shape[0])])

    for _ in range(1, k):
        dist = np.array([min([np.inner(c - x, c - x) for c in centroids]) for x in data])
        probs = dist / dist.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()

        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break

        centroids.append(data[i])

    return np.array(centroids)


def initialize_centroids_kmeans_pp(ds, k, random_state=2):
    #np.random.seed(random_state)
    centroids = [ds[0]]

    for _ in range(1, k):
        dist_sq = np.array([min([np.inner(c - x, c - x) for c in centroids]) for x in ds])
        probs = dist_sq / dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()

        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break

        centroids.append(ds[i])

    return np.array(centroids)

def assign_to_cluster(data, centroids):
    # Assign each data point to the nearest centroid
    distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=0)


def update_centroids(data, assignments):
    # Update the centroids based on the mean of the members of each cluster
    return np.array([data[assignments == i].mean(axis=0) for i in range(np.max(assignments) + 1)])


def mean_intra_distance(data, assignments, centroids):
    # Compute the intra-cluster distance
    return np.sqrt(np.sum((data - centroids[assignments]) ** 2))


def k_means(data, num_centroids, kmeansplusplus=False):
    # centroids initialization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else:
        centroids = initialize_centroids_forgy(data, num_centroids)

    assignments = assign_to_cluster(data, centroids)

    for i in range(100):  # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)

        if np.all(new_assignments == assignments):  # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)
