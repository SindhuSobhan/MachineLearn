# Import libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Import File 
df=pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv')

# Drop empty rows if any
import numpy as np
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)
df.reset_index(inplace=True)

# See data
print(df.head())

# Get Feature matrix by getting numeric fields only
col=[columns for columns in df.columns if df.dtypes[columns] != object]
feat=df[col].values

    
# Normalise data
from sklearn import preprocessing
feat=preprocessing.MinMaxScaler().fit_transform(feat)

# Split data into training and testing sets
Xtrain, Xtest, ytrain, ytest=train_test_split(feat[:,2], feat[:,-1], test_size=0.2, random_state=1)
Xtrain, Xtest, ytrain, ytest= Xtrain.reshape(-1,1), Xtest.reshape(-1,1),\
     ytrain.reshape(-1,1), ytest.reshape(-1,1)

# See Data
import matplotlib.pyplot as plt 
plt.figure()
plt.scatter(feat[:,2], feat[:,-1], color='blue')
plt.title('Data plot')
plt.legend(['Actual Data'])
plt.xlabel('EngineSize')
plt.ylabel('Co2Emission')

# Run SImple Linear regression
from sklearn import linear_model
reg=linear_model.LinearRegression()
reg.fit(Xtrain, ytrain)

# Find Accuracy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
ypred = reg.predict(Xtest)
MAE = mean_absolute_error(ytest, ypred)
MSE = mean_squared_error(ytest, ypred)
r2score = r2_score(ytest, ypred)
print('MAE: %f \nMSE: %f \nR2 Score: %f' % (MAE, MSE, r2score))
print('Coefficients:', reg.coef_)
print('Intercept:', reg.intercept_)

# Plot
xdat=np.linspace(np.min(feat[:,2]), np.max(feat[:,2]), len(feat))
plt.figure()
plt.scatter(feat[:,2], feat[:,-1], c='blue', label='Actual Data')
plt.plot(xdat, reg.predict(xdat.reshape(-1,1)), c='red', linewidth=2, label='Regression Line')
plt.title('Regression plot ('+ str(reg.score(Xtest, ytest))+')')
plt.legend(loc='best')
plt.xlabel('EngineSize')
plt.ylabel('Co2Emission')
plt.axis('tight')

# Show plots
plt.show()