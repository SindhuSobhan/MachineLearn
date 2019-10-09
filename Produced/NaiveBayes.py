# Standard includes
import numpy as np
import matplotlib.pyplot as plt
# Useful module for dealing with the Gaussian density
from scipy.stats import norm



def load_and_process_data(filename):
    # Load data set.
    data = np.loadtxt(filename, delimiter=',')
    # Names of features
    featurenames = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash','Magnesium', 'Total phenols', 
                'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 
                'OD280/OD315 of diluted wines', 'Proline']
    # Split 178 instances into training set (trainx, trainy) of size 130 and test set (testx, testy) of size 48
    np.random.seed(0)
    perm = np.random.permutation(178)
    trainx = data[perm[0:130],1:14]
    trainy = data[perm[0:130],0]
    testx = data[perm[130:178], 1:14]
    testy = data[perm[130:178],0]
    
    return trainx, trainy, testx, testy, featurenames




def fit_naive_generative_model(trainx, trainy, features):
    n_class = len(np.unique(trainy))
    n_feat = len(features)
    
    mu = np.zeros((n_class, n_feat))
    var = np.zeros((n_class, n_feat))
    pi = np.zeros((n_class, 1))
    
    for cl in range(n_class):
        for feat in range(n_feat):
            feature = features[feat]
            indices = (trainy == cl+1)
            mu[cl, feat] = np.mean(trainx[indices, feature])
            var[cl, feat] = np.var(trainx[indices, feature])
            
    pi = np.array([np.sum(trainy == 1), np.sum(trainy == 2), np.sum(trainy == 3)]) / float(len(trainy))
    
    return mu, var, pi 




def fit_normal(x, mu, var):
    P = norm.logpdf(x, mu, np.sqrt(var))
    return P
        

    
    
def predict_class(tx, ty, features, mu, var, pi):
    n_feat = len(features)
    n_class = len(np.unique(ty))
    n = tx.shape[0]
    
    P = np.zeros((n, n_class, n_feat))
    Scores = np.zeros((n, n_class))
    
    for i in range(n):
        for cl in range(0, n_class):
            for feat in range(n_feat):
                feature = features[feat]
                P[i, cl, feat] = fit_normal(tx[i, feature], mu[cl, feat], var[cl, feat])
            Scores[i, cl] = np.log(pi[cl]) +  np.sum(P[i, cl, :])
        
    predictions = np.argmax(Scores, axis=1) + 1
    
    return predictions, P, Scores




def accuracy(predictions, ty, features, featurenames):
    errors = np.sum(predictions != ty)
    
    print("Test error using feature(s): ")
    
    for f in features:
        print("'" + featurenames[f] + "'" + " ")
        
    print("Errors: " + str(errors) + "/" + str(len(ty)) + "= " + str(100* ((len(ty) - errors) / len(ty))) + "% Accuracy")
    return errors
    

    
    
def NaiveBayes(file, features, output = False):
    """Function to obtain the Naive Bayes Classifier for a dataset contained in a comma separated text file.
    
    Inputs: 
    1. Filename in .txt format (a string)
    2. Features or number of the columns that contain the desired features
    3. Output if true, will display Errors values, Probabilitites and Scores for all data points, classes and desired features
    
    Output:
    If output is True:
    Error, Probabilities for each class and each feature of each prediction, and Scores for each class for each prediction
    
    Default: (also if output is False)
    Featurenames and Associated Accuracy"""
    
    
    
    n_out = 3
    n_in = 3
    
    try:
        if (len(np.unique(features)) != len(features)) or (np.sign(np.prod(features)) == -1):
            print("Please Enter Unique positive numbers for features")
            return [None]*n_out
        else:
            #Load and Process Data
            trainx, trainy, testx, testy, featurenames = load_and_process_data(file)
            
            #Fit generative model
            mu, var, pi = fit_naive_generative_model(trainx, trainy, features)
    
            #Find predictions for test cases
            predictions, P, Scores = predict_class(testx, testy, features, mu, var, pi)
    
            #Find Accuracy
            errors = accuracy(predictions, testy, features, featurenames)
            
            if output:
                return errors, P, Scores
    
    except TypeError:
        print("Please Enter the feature set as a list of numbers or an 1D array of numbers or check the type of data entered. It must be numerical")
        return [None]*n_out
    
    except IndexError:
        print("Please Enter a valid Index between 0 and the number of features in the dataset")
        return [None]*n_out
            

#if __name__ == "__main__":
 #   fname = input("Enter Filepath of the dataset. Must be a comma separated .txt file \n")
  #  features = [int(input("Enter feature" + str(feat))) for feat in range(int(input("Enter Number of Features \n")))]
   # NaiveBayes(fname,features)
   