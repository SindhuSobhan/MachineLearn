########################################
##......... KMeans Clustering ........##
########################################



# Import Libraries
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class KMeans:
  """
  Finds clusters in a f dimensional dataset using K-Means (Llyod's ALgorithm)
  Input: Number of clusters (only for class declaration)
  """
  def __init__(self, n_clusters = 3):
      """
      Initialise number fo clusters
      """
      self.n_clusters = n_clusters
    
    
    


  def initialise_general_parameters(self, data):
      """
      Initialise the centroids, cluster and cost variables
      Centroid initialised from randomly chosen points from the dataset
      """
      (self.n, self.f) = data.shape
      random_indices = np.random.randint(low = 0, high = self.n, size = (1, self.n_clusters))
      self.centroids =  np.array(data[random_indices.squeeze(), :])
      self.cluster = np.ones(self.n).astype(np.int32) * 100
      self.costs = []
    




  def initialise_recursive_parameters(self, n_recursion):
      """
      Initialise the centroids, cluster and cost variables for recursive estimation of the best fit
      """
      self.rec_cost_best = []
      self.rec_centroids_best =  np.zeros((self.n_clusters, self.f))
      self.rec_clusters_best = np.ones(self.n).astype(np.int32) * 100
    
    


  def fit(self, data, max_iter = 20, recursive = False, n_recursion = 10, tol = 1e-6):
      """
      Fit the data to get the KMeans clusters

      Inputs:
      data -> Data with shape (n_exampels, n_features)
      max_iter -> Maximum number of iterations
      recursive -> False by default. Make True if best fit of n recursions needed
      n_recursion -> 10 by default. Number of resursions to obtain the best estimate
      tol -> Tolerance for the cost function to decide when to stop. 1e-6 by default.
      """
      self.tol = tol

    # Run general Kemans if recursive is false
      if not recursive:
        # Run general KMeans function
          self.general_kmeans_fit(data, max_iter)
        # Print cost, and cluster centroids
          print("\n\n Final Cost:", self.costs[-1], sep = '\n')
          print("\n\n Final Cluster Centroids:", self.centroids, sep = '\n')
        # If number of datapoints is less than 100, also print the cluster indice for each point
          if self.n < 100:
            print("\n\n Final Cluster values:", self.cluster, sep = "\n")

      else:
        # Run recursive KMeans function
          self.recursive_kmeans_fit(data, max_iter, n_recursion)
        # Print best cost and centroids.
          print("\n\n\n Best fit for:")
          print("\n\nCost:", self.rec_cost_best, sep = '\n')
          print("\n\nCentroids:" , self.rec_centroids_best, sep  = '\n')
        # Also print cluster indices if number of datapoints is less than 11
          if self.n < 100:
              print("\n\nClusters:" , self.rec_clusters_best, sep  = '\n')
    
    


  def general_kmeans_fit(self, data, max_iter):
      """
      Function to carry out the Llyod's KMeans (normal) algorithm
      Assume centroids -> Allocate clusters -> Find Cost -> Update Centroids -> Allocate New clusters ...
      """
    # Initialise the required parameters
      self.initialise_general_parameters(data)
    
    # Iterate for the maximum number of iterations
      for itr in range(max_iter):
        # Function for forward prop or cluster indice allocation and cost calculation (through distance between
        # points and corresponding centroids)
          self.forward_prop(data)
        # Print Cost
          print("Cost for iteration " + str(itr + 1) + ": " + str(self.costs[-1]))
        # If stopping criterion is satisfied, exit the loop and do not update centroids. Else, update.
          if self.stopping_criteria():
              break
          else:
              self.update_param(data)
    
    
        
           
  def recursive_kmeans_fit(self, data, max_iter, n_recursion):
      """
      Function to carry out the Recursive Llyod's KMeans (normal) algorithm, and find the best fit
      General Kmeans -> Save clusters, Save Cost, Save Centroids ->  New General KMeans ... -> Best parameters
      """
    # Initialise general and recursive parameters
      self.initialise_general_parameters(data)
      self.initialise_recursive_parameters(n_recursion)

    # Declare temporary recursive variables
      rec_clusters = np.zeros((n_recursion, self.n))
      rec_centroids = np.zeros((n_recursion, self.n_clusters, self.f))
      rec_cost = np.zeros(n_recursion)
    
    # Iterate for number of chosen recursions and save parameters
      for recur in range(n_recursion):
          print("Recursion ", str(recur + 1), ":")
          self.general_kmeans_fit(data, max_iter)
          rec_clusters[recur] = self.cluster
          rec_centroids[recur, ...] = self.centroids
          rec_cost[recur] = (self.costs[-1])
    
    #Find the Best parameter index based on lowest cost
      ind = np.argmin(rec_cost)
    
    # Save the best parameters (Clusters, centroids, cost)
      self.rec_centroids_best = rec_centroids[ind, ...]
      self.rec_clusters_best = rec_clusters[ind, :]
      self.rec_cost_best = rec_cost[ind]
      
  
  
  
  def forward_prop(self, data):
      """
      Function to allocate clusters and find cost
      """
    # Declare cost and the distance vector
      cost = np.zeros(self.n_clusters)
      dist = np.zeros((self.n, self.n_clusters))
    
    # Find distance from each centroid
      for j, c in enumerate(self.centroids):
          dist[:, j] = np.square(np.linalg.norm(data - c , axis = 1))

    # Assign the datapoint a cluster with minimum distance centroid
      self.cluster = np.argmin(dist, axis = 1).astype(np.int32)

    # Find the cost of all points from their respective centroids
      for j, c in enumerate(self.centroids):
          cost[j] = np.sum(np.linalg.norm(data[self.cluster == j] - c , axis = 1) ** 2) 
      self.costs.append(np.sum(cost))
      
      
      

  def update_param(self, data):
      """
      Function to update the cluster centroids by taking the mean of datapoints assigned to that cluster
      """
    # If some cluster (centroid) is not assigned to any datapoint, re-initialise the centroids
      if len(np.unique(self.cluster)) < self.n_clusters:
          print("Cluster initialisation inadequate, Reinitialising..")
          self.initialise_general_parameters(data)
      else:
        # For each cluster, update the cluster centroid
          for c in range(self.n_clusters):
              self.centroids[c, :] = np.mean(data[self.cluster == c, :], axis = 0)
          
          


  def stopping_criteria(self):
      """
      Function to determine when to stop the iteration for a single KMeans recurrence.
      """
      return len(self.costs) > 1 and (abs(self.costs[-2] - self.costs[-1]) < self.tol)
          




# Main Section to run the class
if __name__ == '__main__':
    # Create Dataset
    dataset = make_blobs(n_samples = 10000, n_features =100, centers = 10)
    # Extract only the data and not the cluster indices (stored in dataset[1])
    data = dataset[0]

    # Create KMeans Class with 5 clusters 
    cl = KMeans(10)
    # Fit the created data
    cl.fit(data, recursive = True)

    # Plot the Actual and KMeans cluster side by side
    plt.figure(figsize = (10, 5))

    # Plot actual clusters
    plt.subplot(1,2,1)
    for i in range(len(np.unique(dataset[1]))):
        plt.scatter(dataset[0][dataset[1] == i,0], dataset[0][dataset[1] == i, 1])
        plt.title('Actual Clusters')

    # Plot KMeans clusters
    plt.subplot(1,2,2)
    for i in range(len(np.unique(dataset[1]))):
        try:
            plt.scatter(dataset[0][cl.rec_clusters_best == i, 0], dataset[0][cl.rec_clusters_best == i, 1])
        except:
            plt.scatter(dataset[0][cl.cluster == i, 0], dataset[0][cl.cluster == i, 1])
        plt.title('K-Means Clusters')
    
    plt.tight_layout()  
    plt.show()