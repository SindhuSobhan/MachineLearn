# Read CSV file  
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

# Split data
Xtrain, Xtest, ytrain, ytest = train_test_split(feat[:, 0:len(col)-2], feat[:,-1], test_size=0.3, random_state=1)

# Run regression
reg=lin_reg()
reg.fit(Xtrain, ytrain)

# Predict and accuracy of fit
ypred = reg.predict(Xtest)
MAE = mean_absolute_error(ytest, ypred)
MSE = mean_squared_error(ytest, ypred)
r2score = r2_score(ytest, ypred)
print('MAE: %f \nMSE: %f \nR2 Score: %f' % (MAE, MSE, r2score))
print('Coefficients:', reg.coef_)
print('Intercept:', reg.intercept_)


# Plot
import matplotlib.pyplot as plt 
plt.plot(ytest, c='blue', linewidth=2, label='Ytest')
plt.plot(ypred, c='red', linewidth=2, label='Ypred')
plt.legend(loc='best')
plt.xlabel('Xdata')
plt.ylabel('Co2Emissions')
plt.grid(axis='both')
plt.axis('tight')
plt.show()
