"""ML_A1_Walid Shaker.ipynb
Original file is located at
    https://colab.research.google.com/drive/1-26Rt_Oc5Y_E5fbJ_GSiiltifvOBcH0S"""

"""#### Libraries Installation"""
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
# %matplotlib inline
#matplotlib for figures, inline >> print more beautifully figures

"""### Read Data and Data exploration"""

#Read Data with Pandas
train_data = pd.read_csv('./flight_delay.csv')

#Data exploration
print(train_data.columns) #Columns of train data
print(train_data.shape) #size of train data
print(train_data.head(5))

#Calculating Flight Duration Column 
train_data['Scheduled depature time'] = pd.to_datetime(train_data['Scheduled depature time'])
train_data['Scheduled arrival time'] = pd.to_datetime(train_data['Scheduled arrival time'])
train_data['Flight Duration'] = (train_data['Scheduled arrival time'] - train_data['Scheduled depature time']).dt.total_seconds()/60
print(train_data.head(5))

#statistics about data, count, unique, top, freq
print(train_data.describe())

"""### Trainset splitting and Reset Index"""
train_data.rename(columns = {'Scheduled depature time': 'Scheduled_depature_time'}, inplace = True)
ts = pd.to_datetime('1/1/2018')
train_data['Scheduled_depature_time']  = pd.to_datetime(train_data.Scheduled_depature_time)
train_set = train_data.loc[train_data.Scheduled_depature_time < ts,:]
test_set = train_data.loc[train_data.Scheduled_depature_time > ts,:]

def reset_index(df):
  df = pd.DataFrame(df)
  df = df.reset_index()
  df = df.drop('index', axis=1)
  return df

train_set = reset_index(train_set)
test_set = reset_index(test_set)


x_train = train_set.drop(['Scheduled_depature_time','Scheduled arrival time','Delay'], axis=1)
x_test = test_set.drop(['Scheduled_depature_time','Scheduled arrival time','Delay'], axis=1)
y_train = train_set['Delay']
y_test = test_set['Delay']

#display last 5 elements in x_train to make sure that they all are in 2017. 
print(x_train.tail(5))

#display first 5 elements in x_test to make sure that they all are in 2018. 
print(x_test.head(5))

"""### One-hot-encoding of categorical feature"""
#check types of x_train, types could be categorical or numerical
types = x_train.dtypes
print("Number categorical features:", sum(types=='object'))
print(types)

print("Encoding Categorical Features\n")
from sklearn.preprocessing import OneHotEncoder

print(f'before encoding: x_train size {x_train.shape} and x_test size {x_test.shape}')

def ohe_new_features(df, features_name, encoder):
    new_feats = encoder.transform(df[features_name])
    #create dataframe from encoded features with named columns
    new_cols = pd.DataFrame(new_feats, dtype=int, columns=encoder.get_feature_names(features_name))
    new_df = pd.concat([df, new_cols], axis=1)   
    new_df.drop(features_name, axis=1, inplace=True)
    return new_df

encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
f_names = ['Depature Airport', 'Destination Airport']
encoder.fit(x_train[f_names])
x_train = ohe_new_features(x_train, f_names, encoder)
x_test = ohe_new_features(x_test, f_names, encoder)
print(f'after encoding: x_train size {x_train.shape} and x_test size {x_test.shape}')


"""### Data Imputation There is no need for imputation as there are no missing values as shown below."""
def count_nans(df):
    return np.sum(np.sum(np.isnan(df)))

print(f'Number of missing values in x_train, x_test: {count_nans(x_train.shape)} , {count_nans(x_test)} \nthere is no need for imputation')

"""### Feature Scaling"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


"""### Data Visualization"""
Flight_Duration = x_train[:,0].reshape(-1,1)
x_test = x_test[:,0].reshape(-1,1)
plt.scatter(Flight_Duration, y_train, marker='.', label="Delay")
plt.title('Flight Duration vs Delay (in minutes)')
plt.xlabel('Flight Duration')
plt.ylabel('Delay (in minutes)')
plt.legend(loc="upper right")
plt.xlim([-0.1, 1.1])
plt.show()

"""### Outlier Detection & Removal"""
print('Outlier Detection & Removal\n')
from sklearn.neighbors import LocalOutlierFactor
print(f'Before OutLier Removal : x_train size {Flight_Duration.shape} and y_train size {y_train.shape}')
lof = LocalOutlierFactor()
yhat = lof.fit_predict(Flight_Duration)
#select all rows that are not outliers
mask = yhat != -1
Flight_Duration, y_train = Flight_Duration[mask, :], y_train[mask]
print(f'After OutLier Removal : x_train size {Flight_Duration.shape} and y_train size {y_train.shape}')
print(f'{np.sum(yhat == -1)} data point have been identified as outliers')
plt.scatter(Flight_Duration, y_train, marker='.', label="Delay")
plt.title('Flight Duration vs Delay (in minutes)')
plt.xlabel('Flight Duration')
plt.ylabel('Delay (in minutes)')
plt.legend(loc="upper right")
plt.xlim([-0.1, 1.1])
plt.show()

"""1. Linear Regression: In this regression task we will predict the flight delay in minutes based upon flight duration."""
print("Linear Regression Model\n")
#Build, Train, and Test Linear Regression Model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
regressor = LinearRegression()
regressor.fit(Flight_Duration, y_train) #uses Gradient Descent

print(f"LR Model intercept : {regressor.intercept_}")
print(f"LR Model coefficient : {regressor.coef_}\n")
y_TRe1 = regressor.predict(Flight_Duration) 
y_pred1 = regressor.predict(x_test) 
eval_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred1})
print(eval_df)

#Evaluate model Training error and Test error
print('\nTrain error')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_TRe1))
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_TRe1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_TRe1)))
print('Coefficient of Determination R score:', metrics.r2_score(y_train, y_TRe1))
print('\nTest error')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred1))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred1))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))
print('Coefficient of Determination R score:', metrics.r2_score(y_test, y_pred1))


"""### 2. Polynomial Regression"""
print('Polynomial Regression Model\n')
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures #to convert the original features into their higher order terms 
from sklearn.model_selection import cross_val_score

degrees = [10,15,20]
plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i])
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                          ("linear_regression", linear_regression)])
    #pipeline is to assemble several steps that can be cross-validated together while setting different parameters.     
    pipeline.fit(Flight_Duration.flatten()[:, np.newaxis], y_train) #flatten a matrix to one dimension.

    
    y_TRe2 = pipeline.predict(Flight_Duration)
    y_pred2 = pipeline.predict(x_test)

    #Evaluate the models using cross validation and evaluate model Training error and Test error
    scores = cross_val_score(pipeline, Flight_Duration.flatten()[:, np.newaxis], y_train,scoring="neg_mean_squared_error", cv=7)
    
    print(f'For degree {degrees[i]}')
    print(f'Cross Validation MSE: {-scores.mean()} and std: {scores.std()}')
    print('Train error')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_train, y_TRe2))
    print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_TRe2))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_TRe2)))
    print('Coefficient of Determination R score:', metrics.r2_score(y_train, y_TRe2))
    print('Test error')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred2))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred2))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred2)))
    print('Coefficient of Determination R score:', metrics.r2_score(y_test, y_pred2))
    print('\n')
    plt.plot(x_test, pipeline.predict(x_test.flatten()[:, np.newaxis]),'r', label="Model")
    plt.scatter(Flight_Duration, y_train, edgecolor='b', s=20, label="Samples")
    plt.xlabel('Flight Duration')
    plt.ylabel('Delay (in minutes)')
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(degrees[i], -scores.mean(), scores.std()))
    plt.xlim([0, 1.1])
plt.show()


"""### 3. Regularization : Lasso and Ridge"""
print('Regularization : Lasso Model\n')
from sklearn.linear_model import Lasso, Ridge
lasso = Lasso()
from sklearn.model_selection import train_test_split
#split x_train to x_train set and validation set in order to tune the hyperparameter alpha
X, y = Flight_Duration, y_train
X, x_val, y, y_val = train_test_split(X, y, test_size=0.1, random_state=123)
alphas = [2.2, 2, 1.5, 1.1, 1, 0.3, 0.1,0.05,0.01] #random variables to select the one with the min error.
losses = []
for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    y_pred3 = lasso.predict(x_val)
    mse = metrics.mean_squared_error(y_val, y_pred3)
    losses.append(mse)
plt.plot(alphas, losses)
plt.title("Lasso alpha value selection")
plt.xlabel("alpha")
plt.ylabel("Mean squared error")
plt.show()

best_alpha = alphas[np.argmin(losses)]
print("Best value of alpha:", best_alpha)

lasso = Lasso(best_alpha)
lasso.fit(X, y)
y_TRe3 = lasso.predict(X)
y_pred3 = lasso.predict(x_test)

print('Train error')
print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_TRe3))
print('Mean Squared Error:', metrics.mean_squared_error(y, y_TRe3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_TRe3)))
print('Coefficient of Determination R score:', metrics.r2_score(y, y_TRe3))
print('Test error')
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred3))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred3))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred3)))
print('Coefficient of Determination R score:', metrics.r2_score(y_test, y_pred3))
