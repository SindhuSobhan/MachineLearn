import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class KMeans:
  
  def __init__(self, n_clusters = 3):
    self.n_clusters = n_clusters
    
    
  def initialise_general_parameters(self, data):
    (self.n, self.f) = data.shape
    random_indices = np.random.randint(low = 0, high = self.n, size = (1, self.n_clusters))
    self.centroids =  np.array(data[random_indices.squeeze(), :])
    self.cluster = np.ones(self.n).astype(np.int32) * 100
    self.costs = []


  def initialise_recursive_parameters(self, n_recursion):
    self.rec_cost_best = []
    self.rec_centroids_best =  np.zeros((self.n_clusters, self.f))
    self.rec_clusters_best = np.ones(self.n).astype(np.int32) * 100
    
    

  def fit(self, data, max_iter = 20, recursive = False, n_recursion = 10, tol = 1e-6):
    self.tol = tol

    if not recursive:
        self.general_kmeans_fit(data, max_iter)
        print("\n\n Final Cost:", self.costs[-1], sep = '\n')
        print("\n\n Final Cluster Centroids:", self.centroids, sep = '\n')
        if self.n < 100:
            print("\n\n Final Cluster values:", self.cluster, sep = "\n")

    else:
        self.recursive_kmeans_fit(data, max_iter, n_recursion)
        print("\n\n\n Best fit for:")
        print("\n\nCost:", self.rec_cost_best, sep = '\n')
        print("\n\nCentroids:" , self.rec_centroids_best, sep  = '\n')
        if self.n < 100:
            print("\n\nClusters:" , self.rec_clusters_best, sep  = '\n')
    
    
      
    
  def general_kmeans_fit(self, data, max_iter):
    self.initialise_general_parameters(data)
    
    for itr in range(max_iter):
        self.forward_prop(data)
        print("Cost for iteration " + str(itr + 1) + ": " + str(self.costs[-1]))
        if self.stopping_criteria():
            break
        else:
            self.update_param(data)
    
    
        
        
        
  def recursive_kmeans_fit(self, data, max_iter, n_recursion):
    self.initialise_general_parameters(data)
    self.initialise_recursive_parameters(n_recursion)

    rec_clusters = np.zeros((n_recursion, self.n))
    rec_centroids = np.zeros((n_recursion, self.n_clusters, self.f))
    rec_cost = np.zeros(n_recursion)
    
    for recur in range(n_recursion):
        print("Recursion ", str(recur + 1), ":")
        self.general_kmeans_fit(data, max_iter)
        rec_clusters[recur] = self.cluster
        rec_centroids[recur, ...] = self.centroids
        rec_cost[recur] = (self.costs[-1])
    
    ind = np.argmin(rec_cost)
    
    self.rec_centroids_best = rec_centroids[ind, ...]
    self.rec_clusters_best = rec_clusters[ind, :]
    self.rec_cost_best = rec_cost[ind]
      
  
  
  
  def forward_prop(self, data):
    cost = np.zeros(self.n_clusters)
    dist = np.zeros((self.n, self.n_clusters))
    
    for j, c in enumerate(self.centroids):
        dist[:, j] = np.square(np.linalg.norm(data - c , axis = 1))

    self.cluster = np.argmin(dist, axis = 1).astype(np.int32)

    for j, c in enumerate(self.centroids):
        cost[j] = np.sum(np.linalg.norm(data[self.cluster == j] - c , axis = 1) ** 2) 
    self.costs.append(np.sum(cost))
      
      
      
  def update_param(self, data):
    if len(np.unique(self.cluster)) < self.n_clusters:
        print("Cluster initialisation inadequate, Reinitialising..")
        self.initialise_general_parameters(data)
    else:
        for c in range(self.n_clusters):
            self.centroids[c] = np.mean(data[self.cluster == c, :], axis = 0)
          
          
          
  def stopping_criteria(self):
    return len(self.costs) > 1 and (abs(self.costs[-2] - self.costs[-1]) < self.tol)
          




if __name__ == '__main__':
    dataset = make_blobs(n_samples = 10000, n_features =2, centers = 5)

    data = dataset[0]
    cl = KMeans(5)
    cl.fit(data, recursive = True)

    plt.figure(figsize = (10, 5))
    plt.subplot(1,2,1)
    for i in range(len(np.unique(dataset[1]))):
        plt.scatter(dataset[0][dataset[1] == i,0], dataset[0][dataset[1] == i, 1])
        plt.title('Actual Clusters')

    plt.subplot(1,2,2)
    for i in range(len(np.unique(dataset[1]))):
        plt.scatter(dataset[0][cl.rec_clusters_best == i, 0], dataset[0][cl.rec_clusters_best == i, 1])
        plt.title('K-Means Clusters')
    
    plt.tight_layout()  
    plt.show()
