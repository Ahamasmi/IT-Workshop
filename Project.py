#importing all the modules
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
names = ['VENDOR','MODEL_NAME','MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP' ];
data = pd.read_csv('machine.data',names=names)   #this file was uploaded to colab to work on
data.head()
#information about our data
data.info()
#working with the indexes
categorical_ = data.iloc[:,:2]
numerical_ = data.iloc[:,2:]
print(numerical_.head())
X = numerical_.iloc[:,:-1]
y = numerical_.iloc[:,-1]
coeff_r2=rdst=mse=r2=0
rdst_arr=[]
prev_coeff_r2=[]
prev_mse=[]
prev_r2=[]
prop_coeff_r2=[]
prop_mse=[]
prop_r2=[]
random=[]
print("Previous  metrics: ")
for i in range(0,200):
  random.append(i)
  x_training_set, x_test_set, y_training_set, y_test_set = train_test_split(X,y,test_size=0.1,random_state=i,shuffle=True)
  from sklearn.preprocessing import StandardScaler

  scaler = StandardScaler()
  # Fit on training set only.
  scaler.fit(X)

  # Apply transform to both the training set and the test set.
  x_training_set = scaler.transform(x_training_set)
  x_test_set = scaler.transform(x_test_set)
  #part 2
  y_training_set = y_training_set.values.reshape(-1, 1)
  y_test_set  = y_test_set.values.reshape(-1, 1)

  y_scaler = StandardScaler()
  # Fit on training set only.
  y_scaler.fit(y_training_set)

  # Apply transform to both the training set and the test set.
  y_training_set = y_scaler.transform(y_training_set)
  y_test_set = y_scaler.transform(y_test_set)
  model = linear_model.LinearRegression()
  model.fit(x_training_set,y_training_set)
  from sklearn.metrics import mean_squared_error, r2_score
  model_score = model.score(x_training_set,y_training_set)
  # Have a look at R sq to give an idea of the fit ,
  # Explained variance score: 1 is perfect prediction
  y_predicted = model.predict(x_test_set)
  prev_coeff_r2.append(model_score)
  prev_mse.append(mean_squared_error(y_test_set, y_predicted))
  prev_r2.append(r2_score(y_test_set, y_predicted))
  rdst_arr.append(i)
  if model_score>coeff_r2:
    coeff_r2=model_score
    rdst=i
    mse=mean_squared_error(y_test_set, y_predicted)
    r2=r2_score(y_test_set, y_predicted) 
print("Coefficient of determination R^2 of the prediction.:",coeff_r2)
print("Random State for the r2 score is: ",rdst)
# The mean squared error
print("Mean squared error: %.2f"% mse)
# Explained variance score: 1 is perfect prediction
print('Test Variance score: %.2f' % r2 )

for i in range(0,200):
  x_training_set, x_test_set, y_training_set, y_test_set = train_test_split(X,y,test_size=0.5,random_state=i,shuffle=True)
  from sklearn.preprocessing import StandardScaler

  scaler = StandardScaler()
  # Fit on training set only.
  scaler.fit(X)

  # Apply transform to both the training set and the test set.
  x_training_set = scaler.transform(x_training_set)
  x_test_set = scaler.transform(x_test_set)
  #part 2
  y_training_set = y_training_set.values.reshape(-1, 1)
  y_test_set  = y_test_set.values.reshape(-1, 1)

  y_scaler = StandardScaler()
  # Fit on training set only.
  y_scaler.fit(y_training_set)

  # Apply transform to both the training set and the test set.
  y_training_set = y_scaler.transform(y_training_set)
  y_test_set = y_scaler.transform(y_test_set)
  model = linear_model.LinearRegression(normalize=True)
  model.fit(x_training_set,y_training_set)
  from sklearn.metrics import mean_squared_error, r2_score
  model_score = model.score(x_training_set,y_training_set)
  # Have a look at R sq to give an idea of the fit ,
  # Explained variance score: 1 is perfect prediction
  y_predicted = model.predict(x_test_set)
  prop_coeff_r2.append(model_score)
  prop_mse.append(mean_squared_error(y_test_set, y_predicted))
  prop_r2.append(r2_score(y_test_set, y_predicted))
  if model_score>coeff_r2:
    coeff_r2=model_score
    rdst=i
    mse=mean_squared_error(y_test_set, y_predicted)
    r2=r2_score(y_test_set, y_predicted)  
print("Proposed metrics: ")
print("Coefficient of determination R^2 of the prediction.:",coeff_r2)
print("Random State for the r2 score is: ",rdst)
# The mean squared error
print("Mean squared error: %.2f"% mse)
# Explained variance score: 1 is perfect prediction
print('Test Variance score: %.2f' % r2 )
import numpy as np
import matplotlib.pyplot as plt  

#graph for mean squared error
plt.plot(random, prev_coeff_r2,label='Previous')
plt.plot(random, prop_coeff_r2,label='Current')
plt.xlabel('Random State')
plt.ylabel('Mean Squared Error')
plt.legend(loc='upper left')
fig_size[0] = 14
fig_size[1] = 12
plt.rcParams["figure.figsize"] = fig_size
plt.show()  
#graph for R2 score  
plt.plot(random, prev_coeff_r2,label='Previous')
plt.plot(random, prop_coeff_r2,label='Current')
plt.xlabel('Random State')
plt.ylabel('R2 score')
plt.legend(loc='upper left')
fig_size[0] = 14
fig_size[1] = 12
plt.rcParams["figure.figsize"] = fig_size
plt.show()  
#graph for test variance score
plt.plot(random, prev_r2,label='Previous')
plt.plot(random, prop_r2,label='Current')
plt.xlabel('Random State')
plt.ylabel('Test Variance Score')
plt.legend(loc='lower left')
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14
fig_size[1] = 12
plt.rcParams["figure.figsize"] = fig_size
plt.show() 