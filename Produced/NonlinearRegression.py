# Import Data
import pandas as pd 
df=pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/china_gdp.csv')

# See data
print(df.head())

# Remove empty data
import numpy as np
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# make feature set
feat = df.values

# Normalise feature set
from sklearn.preprocessing import MinMaxScaler
feat=MinMaxScaler().fit_transform(feat)

# Check data visually
import matplotlib.pyplot as plt 
plt.figure()
plt.scatter(feat[:,0], feat[:,1], label='GDP')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.legend(loc='best')

# Split data
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(feat[:,0], feat[:,1], test_size=0.3, random_state=1)

# Define Sigmoid
import numpy as np 
def sigmoid(X, beta1, beta2):
    y = 1/(1+np.exp(-beta1*(X-beta2)))
    return y

# Check Sigmoid curve
plt.figure()
beta1, beta2, X = 1, 2, np.linspace(0, 10, 100) 
plt.plot(X, sigmoid(X, beta1, beta2), c='blue', label='Sigmoid: beta=[1,2]')
plt.legend(loc='best')

# Run Regression
from scipy.optimize import curve_fit
popt, pcov=curve_fit(sigmoid, Xtrain, ytrain)

# Display Parameters
print('Parameters are:', popt)

# Accuracy
from sklearn.metrics import r2_score
ypred = sigmoid(Xtest, *popt)
print('MAE: ', np.mean(np.absolute(ytest-ypred)))
print('MSE: ', np.mean((ypred-ytest)**2))
print('R2-Score: ', r2_score(ytest, ypred))

# Display plot
plt.figure()
plt.scatter(feat[:,0], feat[:,1], c='blue', label='Actual')
plt.plot(feat[:,0], sigmoid(feat[:,0], *popt), '-r', linewidth=2, label='Fitted')
plt.xlabel('Normalised Year')
plt.ylabel('Normalised GDP')
plt.legend(loc='best')
plt.show()