########################################
##...... Variational AutoEncoder .....##
########################################



# Import Libraries
from __future__ import print_function, division
from builtins import range

import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt

from keras.datasets import mnist 

st  = None
try:
    # If this module still exists, use it
    st = tf.contrib.bayesflow.stochastic_tensor
except:
    pass


# Save distributions in a shortened form
Normal = tf.contrib.distributions.Normal
Bernoulli = tf.contrib.distributions.Bernoulli



# Dense Layer object
class DenseLayer(object):
    # Define layer parameters 
    def __init__(self, L1, L2, A = tf.nn.relu):
         self.W = tf.Variable(tf.random_normal(shape = (L1, L2)) * 2 / np.sqrt(L1))
         self.b = tf.Variable(np.zeros(L2).astype(np.float32))
         self.A = A

    # Compute forward transfer through one layer 
    def forward(self, X):
        return self.A(tf.matmul(X, self.W) + self.b)



# Class for Variational Autoencoder
class Variational_Autoencoder:

    """
    Class to execute a Variational Autoencoder.

    Class definition requires two inputs:
    D -> The number of features in the input
    hidden_layer_sizes -> The number of neurons or sizes of all hidden layers in the encoder step

    Once defined, Mnist or other dataset can be used to execute the fit function to initiate the training.

    Please read individual function definitions for more information.
    """
    
    def __init__(self, D, hidden_layer_sizes):

        """
        Defines the necessary variables, layers, cost functions and other initialisers. 
        """
        self.X = tf.placeholder(tf.float32, shape = (None, D))


        #... ENCODER ...#
        # Define the encoder layers and compute the layer values to find the latent variable distribution.
        # The means and standard deviations are obtained after the last layer.

        self.encoder_layers = []
        L_in = D

        for L_out in hidden_layer_sizes[:-1]:
            h = DenseLayer(L_in, L_out)
            self.encoder_layers.append(h)
            L_in = L_out

        L = hidden_layer_sizes[-1]

        h = DenseLayer(L_in, 2 * L, A = lambda x: x)
        self.encoder_layers.append(h)

        current_layer = self.X
        for layer in self.encoder_layers:
            current_layer = layer.forward(current_layer)

        self.means = current_layer[:, :L]
        self.stddev = tf.nn.softplus(current_layer[:, L:]) + 1e-6
        
        # Find the distributions
        if st is None:
            standard_normal = Normal(
                loc = np.zeros(L, dtype=np.float32),
                scale = np.ones(L, dtype=np.float32)
            )
            e = standard_normal.sample(tf.shape(self.means)[0])
            self.Z = e * self.stddev + self.means
        else:
            with st.value_type(st.SampleValue()):
                self.Z = st.StochasticTensor(Normal(loc=self.means, scale=self.stddev))
        

        #... DECODER ...#
        # Define the decoder layers and compute the layer values to find the posterior and prior distributions
        
        self.decoder_layers = []
        L_in = L

        for L_out in reversed(hidden_layer_sizes[ :-1]):
            h = DenseLayer(L_in, L_out)
            self.decoder_layers.append(h)
            L_in = L_out

        h = DenseLayer(L_in, D, A = lambda x : x)
        self.decoder_layers.append(h)

        current_layer = self.Z
        for layer in self.decoder_layers:
            current_layer = layer.forward(current_layer)
        
        logits = current_layer
        posterior_predictive_logits = logits

        self.X_hat_distribution = Bernoulli(logits = logits)

        self.posterior_predictive = self.X_hat_distribution.sample()
        self.posterior_predictive_probs = tf.nn.sigmoid(logits)

        #... Prior Predictive ...#
        standard_normal = Normal(
            loc = np.zeros(L, dtype = np.float32),
            scale = np.ones(L, dtype = np.float32)
        )

        Z_std = standard_normal.sample(1)
        current_layer = Z_std
        for layer in self.decoder_layers:
            current_layer = layer.forward(current_layer)
        logits = current_layer

        prior_predictive_dist = Bernoulli(logits = logits)
        self.prior_predictive = prior_predictive_dist.sample()
        self.prior_predictive_probs = tf.nn.sigmoid(logits)

        self.Z_input = tf.placeholder(tf.float32, shape = (None, L))
        current_layer = self.Z_input
        for layer in self.decoder_layers:
            current_layer = layer.forward(current_layer)
        logits = current_layer
        self.prior_predictive_from_input_probs = tf.nn.sigmoid(logits)

        #... Cost ...#
        if st is None:
                kl = -1 * tf.log(self.stddev) + 0.5*(self.stddev**2 + self.means**2) - 0.5
                kl = tf.reduce_sum(kl, axis=1)
        else:
            kl = tf.reduce_sum(
                tf.contrib.distributions.kl_divergence(
                self.Z.distribution, standard_normal
                ),
                1  
            )
          
        expected_log_likelihood = tf.reduce_sum(
            self.X_hat_distribution.log_prob(self.X),
            1
            )

        self.ELBO =  tf.reduce_sum(expected_log_likelihood - kl)
        self.train_op = tf.train.RMSPropOptimizer(learning_rate = 0.001).minimize(-self.ELBO)

        self.init_op = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init_op)
    

    def get_mnist(self):
        """ 
        To obtain the Mnist dataset in the required format, simply execute the get_mnist() function provided
        in the class itself. For other datasets, please ensure that the dataset is arranged in 
        (Number_of_examples, features) format. 
        """
        (X, Y), (_,_) = mnist.load_data()
        X = X.reshape(60000, 784)
        X = X/ 255.0 # data is from 0..255
        return X, Y

    

    def fit(self, X, epochs = 30, batch_size = 64):
        """
        Fit the data by training the model on it for a specified number of epochs and batch_size. 
        It is best if the data is normalised befor it is fed in for training. 
        """
        costs = []
        n_batches = len(X) // batch_size
        print("Number of Batches:" , n_batches)

        for i in range(epochs):
            print("Epoch ", i, ": " )
            np.random.shuffle(X)
            
            for j in range(n_batches):
                batch = X[j * batch_size : (j+1) * batch_size]
                _, c, = self.sess.run((self.train_op, self.ELBO), feed_dict = {self.X : batch})
                c = c/ batch_size
                costs.append(c)
                if j%100 == 0:
                    print("Iter: %d, Cost: %0.3f" % (j, c))
        plt.plot(costs)
        plt.show()


    def transform(self, X):
        """
        Calculate the mean obtained as the final layer of the encoder.
        """
        return self.sess.run(
            self.means,
            feed_dict = {self.X: X}
        )


    def prior_predictive_with_input(self, Z):
        """
        Calculate the prior probability for a given distributuion Z
        """
        return self.sess.run(
            self.prior_predictive_from_input_probs, 
            feed_dict = {self.Z: Z}
        )


    def posterior_predicitve_sample(self, X):
        """
        Obtain posterior predictive sample for given X
        """
        return self.sess.run(self.posterior_predictive, feed_dict = {self.X : X})

    
    def prior_predictive_sample_with_probs(self):
        """
        Obtain prior predictive sample with standard normal as the prior
        """
        return self.sess.run((self.prior_predictive, self.prior_predictive_probs))




#... Executes first when file runs ...#
if __name__ == '__main__':
    # Create Vae class object
    vae = Variational_Autoencoder(784, [200, 100])

    # Get data
    X, Y = vae.get_mnist()

    # Convert data to binary
    X = (X > 0.5).astype(np.float32)

    # Fit data to model
    vae.fit(X)

    # Display samples and original images
    done = False
    while not done:
        i = np.random.choice(len(X))
        x = X[i]
        im = vae.posterior_predicitve_sample([x]).reshape(28, 28)
        plt.subplot(1, 2, 1)
        plt.imshow(x.reshape(28, 28), cmap = 'gray')
        plt.title("Original")

        plt.subplot(1, 2, 2)
        plt.imshow(im, cmap = 'gray')
        plt.title("Sample")

        plt.show()

        ans = input("Generate Another")
        if ans and ans[0] in ('n' or 'N'):
            done  = True

    #Display Prior samples and probs
    done = False
    while not done:
        im, probs = vae.prior_predictive_sample_with_probs()
        im = im.reshape(28, 28)
        probs = probs.reshape(28, 28)
        plt.subplot(1, 2, 1)
        plt.imshow(im, cmap = 'gray')
        plt.title("Prior Predictive Sample")

        plt.subplot(1, 2, 2)
        plt.imshow(probs, cmap = 'gray')
        plt.title("Prior Predicitve probs")

        plt.show()

        ans = input("Generate Another")
        if ans and ans[0] in ('n' or 'N'):
            done  = True