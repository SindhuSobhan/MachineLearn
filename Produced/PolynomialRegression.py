# Import the data
import pandas as pd 
df=pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/FuelConsumptionCo2.csv')

# Find empty rows and delete them
import numpy as np
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# Extract Feature set
col=[column for column in df.columns if df.dtypes[column] != object]
feat=df[col].values

# import libraries regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lin_reg 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Scale Data
feat=StandardScaler().fit_transform(feat)

# Fit polynomial features
from sklearn.preprocessing import PolynomialFeatures
xdata = PolynomialFeatures(degree=3).fit_transform(feat[:,1].reshape(-1,1))
ydata=feat[:,-1].reshape(-1,1)

# Split data
Xtrain, Xtest, ytrain, ytest = train_test_split(xdata, ydata, test_size=0.3, random_state=1)

# Fit regression
reg=lin_reg()
reg.fit(Xtrain, ytrain)

# Accuracy metrics
ypred = reg.predict( Xtest)
MAE = mean_absolute_error(ytest, ypred)
MSE = mean_squared_error(ytest, ypred)
r2score = r2_score(ytest, ypred)
print('MAE: %f \nMSE: %f \nR2 Score: %f' % (MAE, MSE, r2score))
print('Coefficients:', reg.coef_)
print('Intercept:', reg.intercept_)

# Plot curves
import matplotlib.pyplot as plt 
import numpy as np 
x=np.linspace(np.min(feat[:,1]), np.max(feat[:,1]), feat.shape[0])
X = PolynomialFeatures(degree=3).fit_transform(x.reshape(-1,1))
plt.scatter(feat[:,1], feat[:,-1], c='blue', label='Actual Data')
plt.plot( x, reg.predict(X), c='red', linewidth='2', label='Regression Line')
plt.legend(loc='best')
plt.xlabel('Engine Size')
plt.ylabel('Co2 Emissions')
plt.axis('tight')
plt.show()
