import numpy as np 
from scipy.stats import multivariate_normal as mvn 
import tensorflow as tf 
from keras.datasets import mnist,fashion_mnist
import matplotlib.pyplot as plt 

class BayesClassifier():

    def __init__(self):
        self.k = 0
        self.gaussian = []

    def load_and_process(self, data = 'mnist'):
        if data == 'mnist':
            (X, Y),  (_, _) = mnist.load_data()
            (n,h,w) = X.shape
            X = X.reshape(n, h * w) 
            return X, Y
        if data == 'fashion_mnist':
            (X, Y),  (_, _) = fashion_mnist.load_data()
            (n,h,w) = X.shape
            X = X.reshape(n, h * w) 
            return X, Y

    def fit(self, trainx, trainy):
        self.k = len(set(trainy))

        for cl in range(self.k):
            ind = (trainy == cl)
            X = trainx[ind, :]
            cov = np.cov(X, rowvar = False, bias = True)
            m = np.mean(X, axis = 0)
            self.gaussian.append({"m": m, "cov": cov})


    def sample_for_y(self, y):
        g = self.gaussian[y]
        return mvn.rvs(mean = g["m"], cov = g["cov"])

    def sample(self):
        y = np.random.randint(0, self.k)
        return self.sample_for_y(y)



if __name__ == "__main__":
    clf = BayesClassifier()
    trainx, trainy = clf.load_and_process(data = 'fashion_mnist')
    clf.fit(trainx, trainy)

    plt.figure(figsize= [20, 20])

    for cl in range(clf.k):
        sample = clf.sample_for_y(cl).reshape(28, 28)
        mean = clf.gaussian[cl]["m"].reshape(28, 28)

        plt.subplot(10,3,3*cl+1)
        plt.imshow(sample, cmap = 'gray')
        plt.title('Sample')
        plt.subplot(10,3,3*cl + 2)
        plt.imshow(mean, cmap = 'gray')
        plt.title('Mean')

        random_sample = clf.sample().reshape(28,28)
        plt.subplot(10,3,3*cl+3)
        plt.imshow(random_sample, cmap = 'gray')
        plt.title('Random Sample')
    
    plt.show()
    plt.tight_layout()