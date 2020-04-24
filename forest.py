import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import pickle

housing = pd.read_csv("data.csv")
print(housing.head())

#housing.describe()
#print(housing.shape)
#print(housing.columns)
#print(housing.isnull().sum())
#print(housing.info())

#%matplotlib inline
#housing.hist(bins=50,figsize=(20,30),color='green')
#plt.show()


#Looking for Corelation
#corr_matrix = housing.corr()
#corr_percentage = (corr_matrix['MEDV'].sort_values(ascending=False))*100
#corr_percentage

#fig, ax = plt.subplots(figsize=(18,10))
#sns.heatmap(corr_matrix, annot=True, fmt='.0%')
#plt.show()

#StratifiedShuffleSplit SO THAT TO AVOID COMPLETE 1 OR 0 IN CHAS while in test and train data
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.20,random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    X_train = housing.loc[train_index]
    X_test = housing.loc[test_index]

X = housing.drop('MEDV', axis=1)
Y = housing['MEDV'] 
#print(X.shape)
#print(Y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.20, random_state=5)
print("X_train.shape =",X_train.shape)
print("X_test.shape =",X_test.shape)
print("Y_train.shape =",Y_train.shape)
print("Y_test.shape =",Y_test.shape)

#To scale data from 0 to 1 applying preprocessing MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test =  scaler.fit_transform(X_test)


#Linear Regression
#from sklearn.linear_model import LinearRegression
#LModel = LinearRegression()
#LModel.fit(X_train,Y_train)
#LModel.score(X_train,Y_train)

#housing_predictions = LModel.predict(X_test) #X_test
#LModel_mse = mean_squared_error(Y_test, housing_predictions) #y_test, Y_pred
#LModel_rmse = np.sqrt(LModel_mse)
#LModel_r2 =  r2_score(Y_test, housing_predictions) #y_test, Y_pred

#print("MSE score ",LModel_mse)
#print("RMSE Score ",LModel_rmse)
#print("R2 Score ",LModel_r2)

#Ridge Regression(Regularised Linear Regression)
#from sklearn.linear_model import Ridge
#from sklearn.model_selection import GridSearchCV

#ridge = Ridge()
#parameters ={'alpha': [1,5,10,20, 30,35,40,45,50,55,100]}
#ridge_regressor = GridSearchCV(ridge, parameters, scoring ="neg_mean_squared_error", cv=10)
#ridge_regressor.fit(X_train,Y_train)

#print(ridge_regressor.best_params_)
#print(ridge_regressor.best_score_)

#housing_predictions_ridge = ridge_regressor.predict(X_test) #X_test
#ridge_LModel_mse = mean_squared_error(Y_test, housing_predictions_ridge) #y_test, Y_pred
#ridge_Model_rmse = np.sqrt(ridge_LModel_mse)
#ridge_Model_r2 =  r2_score(Y_test, housing_predictions_ridge) #y_test, Y_pred

#print("MSE score ",ridge_LModel_mse)
#print("RMSE Score ",ridge_Model_rmse)
#print("R2 Score ",ridge_Model_r2)

#Lasso Regression(Regularised Linear Regression)
#from sklearn.linear_model import Lasso
#from sklearn.model_selection import GridSearchCV

#lasso = Lasso()
#parameters ={'alpha': [1,5,10,20, 30,35,40,45,50,55,100]}
#lasso_regressor = GridSearchCV(ridge, parameters, scoring ="neg_mean_squared_error", cv=10)
#lasso_regressor.fit(X_train,Y_train)

#print(lasso_regressor.best_params_)
#print(lasso_regressor.best_score_)

#housing_predictions_lasso = lasso_regressor.predict(X_test) #X_test
#lasso_LModel_mse = mean_squared_error(Y_test, housing_predictions_lasso) #y_test, Y_pred
#lasso_Model_rmse = np.sqrt(lasso_LModel_mse)
#lasso_Model_r2 =  r2_score(Y_test, housing_predictions_lasso) #y_test, Y_pred

#print("MSE score ",lasso_LModel_mse)
#print("RMSE Score ",lasso_Model_rmse)
#print("R2 Score ",lasso_Model_r2)

#Elastic Net(Regularised Linear regression)
#from sklearn.linear_model import ElasticNet
#from sklearn.model_selection import GridSearchCV

#elastic_net = ElasticNet()
#parameters ={'alpha': [1,5,10,20, 30,35,40,45,50,55,100]}
#elastic_net = GridSearchCV(ridge, parameters, scoring ="neg_mean_squared_error", cv=10)
#elastic_net.fit(X_train,Y_train)

#housing_predictions_elastic_net = elastic_net.predict(X_test) #X_test
#elastic_LModel_mse = mean_squared_error(Y_test, housing_predictions_elastic_net) #y_test, Y_pred
#elastic_Model_rmse = np.sqrt(elastic_LModel_mse)
#elastic_Model_r2 =  r2_score(Y_test, housing_predictions_elastic_net) #y_test, Y_pred

#print("MSE score ",elastic_LModel_mse)
#print("RMSE Score ",elastic_Model_rmse)
#print("R2 Score ",elastic_Model_r2)


#Support Vector Machine
#from sklearn.svm import SVR
#svr = SVR(C=1.0, epsilon=0.2,degree=4,gamma='scale')
#svr.fit(X_train,Y_train)
#svr.score(X_train,Y_train)

#housing_predictions_svr = svr.predict(X_test) #X_test
#svr_Model_mse = mean_squared_error(Y_test, housing_predictions_svr) #y_test, Y_pred
#svr_Model_rmse = np.sqrt(svr_Model_mse)
#svr_Model_r2 =  r2_score(Y_test, housing_predictions_svr) #y_test, Y_pred

#print("MSE score ",svr_Model_mse)
#print("RMSE Score ",svr_Model_rmse)
#print("R2 Score ",svr_Model_r2)


#DecisionTreeRegressor
#from sklearn.tree import DecisionTreeRegressor
#tree = DecisionTreeRegressor(max_depth=10)
#tree.fit(X_train,Y_train)
#tree.score(X_train,Y_train)

#housing_predictions_tree = tree.predict(X_test) #X_test
#tree_Model_mse = mean_squared_error(Y_test, housing_predictions_tree) #y_test, Y_pred
#tree_Model_rmse = np.sqrt(tree_Model_mse)
#tree_Model_r2 =  r2_score(Y_test, housing_predictions_tree) #y_test, Y_pred

#print("MSE score ",tree_Model_mse)
#print("RMSE Score ",tree_Model_rmse)
#print("R2 Score ",tree_Model_r2)


#GradientBoostingRegressor
#from sklearn.ensemble import GradientBoostingRegressor
#gradientregressor = GradientBoostingRegressor(max_depth=2,n_estimators=3,learning_rate=1)
#gradientregressor.fit(X_train,Y_train)
#gradientregressor.score(X_train,Y_train)

#housing_predictions_gradientregressor = gradientregressor.predict(X_test) #X_test
#gradientregressor_Model_mse = mean_squared_error(Y_test, housing_predictions_gradientregressor) #y_test, Y_pred
#gradientregressor_Model_rmse = np.sqrt(gradientregressor_Model_mse)
#gradientregressor_Model_r2 =  r2_score(Y_test, housing_predictions_forest) #y_test, Y_pred

#print("MSE score ",gradientregressor_Model_mse)
#print("RMSE Score ",gradientregressor_Model_rmse)
#print("R2 Score ",gradientregressor_Model_r2)


#Hyper Parameter Tuning
#from sklearn.model_selection import GridSearchCV

#LR = {'learning_rate': [0.15,0.1,0.10,0.5],'n_estimators': [100,150,200,250]}

#tuning = GridSearchCV(estimator=GradientBoostingRegressor(),param_grid=LR,scoring='r2')

#tuning.fit(X_train,Y_train)
#print(tuning.best_params_)
#print(tuning.best_score_)


#AdaBoostRegressor
#from sklearn.ensemble import AdaBoostRegressor
#AdaModel = AdaBoostRegressor(n_estimators=100,learning_rate=1)
#AdaModel.fit(X_train,Y_train)
#AdaModel.score(X_train,Y_train)

#housing_predictions_AdaModel = AdaModel.predict(X_test) #X_test
#AdaModel_Model_mse = mean_squared_error(Y_test, housing_predictions_AdaModel) #y_test, Y_pred
#AdaModel_Model_rmse = np.sqrt(AdaModel_Model_mse)
#AdaModel_Model_r2 =  r2_score(Y_test, housing_predictions_AdaModel) #y_test, Y_pred

#print("MSE score ",AdaModel_Model_mse)
#print("RMSE Score ",AdaModel_Model_rmse)
#print("R2 Score ",AdaModel_Model_r2)

#Hyper Parameter Tuning
#from sklearn.model_selection import GridSearchCV

#LR = {'learning_rate': [0.15,0.18,0.1,0.10,0.5],'n_estimators': [100,150,200,250]}

#tuning_model = GridSearchCV(estimator=GradientBoostingRegressor(),param_grid=LR,scoring='r2')

#tuning_model.fit(X_train,Y_train)
#print(tuning_model.best_params_)
#print(tuning_model.best_score_)


#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators = 30,min_samples_split=2,max_depth=25)
forest.fit(X_train,Y_train)
print(forest.score(X_train,Y_train))

housing_predictions_forest = forest.predict(X_test) #X_test
forest_Model_mse = mean_squared_error(Y_test, housing_predictions_forest) #y_test, Y_pred
forest_Model_rmse = np.sqrt(forest_Model_mse)
forest_Model_r2 =  r2_score(Y_test, housing_predictions_forest) #y_test, Y_pred

print("MSE score ",forest_Model_mse)
print("RMSE Score ",forest_Model_rmse)
print("R2 Score ",forest_Model_r2)


#Artificial Neuro Network
#import keras
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation
#from keras import metrics

#first input and first hidden layer
#model = Sequential()
#model.add(Dense(20,input_dim=13,activation='relu'))
#second hidden layer
#model.add(Dense(10,activation='relu'))
#third hidden layer
#model.add(Dense(1,activation='linear'))
#compile ANN
#model.compile(optimizer='Adam',
 #             loss='mean_squared_error',
 #             metrics=['accuracy'])

#fit and display summery
#model.fit(X_train, Y_train,
#          epochs=1000, verbose=1)

#model.summary()

#Testing the test dataset
#y_predKM = model.predict(X_test)
#score = model.evaluate(X_test,Y_test,verbose=0)
#print(score[0])


#Dump the model
pickle.dump(forest, open('HousePrediction.pkl','wb'))
#reload the model
model = pickle.load(open('HousePrediction.pkl','rb'))