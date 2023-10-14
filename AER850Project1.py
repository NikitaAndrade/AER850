import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, f1_score, confusion_matrix

#-----2.1 Data Processing-----
data_path = 'data\Project 1 Data.csv'
df = pd.read_csv(data_path)
df = df.replace(0, -1) #encode ths Zeros to a recognizable value as NaN values are not supported for Regression Models 
df = df.dropna()
df = df.reset_index(drop=True) 
train_y = df['Step']
df_X = df.drop(columns = ["Step"])

#scaling the variables 
my_scaler = StandardScaler()
my_scaler.fit(df_X.iloc[:,:])
scaled_data = my_scaler.transform(df_X.iloc[:,:])
scaled_data_df = pd.DataFrame(scaled_data, columns=df_X.columns[:])
train_X = scaled_data_df
columns_list = train_X.columns.tolist()

#-----2.2 Data Visualization-----
attributes = ["X", "Y", "Z","Step"]
pd.plotting.scatter_matrix(df[attributes], figsize=(12, 8))
plt.savefig("C:/Users/Owner/Documents/Python/GitHub/Plots/Data Visualization.png")
plt.show()

#Using the Data Visualization method of scatter plot matrix, no corelation is found between the variables. 

#-----2.3 Corelation Analysis-----
#Pearsons Method
correlation_matrix = df.corr()
print(correlation_matrix)
plt.title('Pearson Correlation Matrix')
sns.heatmap(np.abs(correlation_matrix), annot=True)
plt.savefig("C:/Users/Owner/Documents/Python/GitHub/Plots/Pearson's Correlation Matrix.png")
colormap = plt.get_cmap()
colormap_name = colormap.name
print(colormap_name)
plt.show()
corr1 = np.corrcoef(df['X'], df['Step'])
print("X correlation with Step is: ", round(corr1[0,1],2))
corr1 = np.corrcoef(df['Y'], df['Step'])
print("Y correlation with Step is: ", round(corr1[0,1],2))
corr1 = np.corrcoef(df['Z'], df['Step'])
print("Z correlation with Step is: ", corr1[0,1])


#-----2.4 Classification Model Development/Engineering-----
#Model 1: Linear Regressor 
model1 = LinearRegression()
model1.fit(train_X, train_y)

model1_prediction = model1.predict(train_X)
from sklearn.metrics import mean_absolute_error
model1_train_mae = mean_absolute_error(model1_prediction, train_y)
print("Model 1 training MAE is: ", round(model1_train_mae,2))
#Model 1 training MAE is:  0.76.


#Model 2: Random Forest Regressor 
# model2 = RandomForestRegressor(n_estimators=10, random_state=16) #Pre Grid Search
model2 = RandomForestRegressor(max_depth = None, max_features = 'sqrt', min_samples_leaf= 1, min_samples_split= 10, n_estimators=30, random_state=16)
model2.fit(train_X, train_y)
model2_predictions = model2.predict(train_X)
model2_train_mae = mean_absolute_error(model2_predictions, train_y)
print("Model 2 training MAE is: ", round(model2_train_mae,2))
#Model 2 training MAE is:  0.01


#Model 3: Decision Tree Regressor
# model3= DecisionTreeRegressor() #Pre Grid Search
model3 = DecisionTreeRegressor(criterion='absolute_error', max_depth=10, max_features='log2', max_leaf_nodes=None, min_impurity_decrease=0.0, min_samples_leaf=4, min_samples_split=5)
model3.fit(train_X, train_y)
model3_predictions = model3.predict(train_X)
model3_train_mae = mean_absolute_error(model3_predictions, train_y)
print("Model 3 training MAE is: ", round(model3_train_mae,2))
#Model 3 training MAE is: 0.0

#Cross Validation
# Perform k-fold cross-validation for Model 1
scores_model1 = cross_val_score(model1, train_X, train_y, cv=5, scoring='neg_mean_absolute_error')
mae_model1 = -scores_model1.mean()
print("Model 1 Mean Absolute Error (CV):", round(mae_model1, 2))

# Perform k-fold cross-validation for Model 2
scores_model2 = cross_val_score(model2, train_X, train_y, cv=5, scoring='neg_mean_absolute_error')
mae_model2 = -scores_model2.mean()
print("Model 2 Mean Absolute Error (CV):", round(mae_model2, 2))

# Perform k-fold cross-validation for Model 3
scores_model3 = cross_val_score(model3, train_X, train_y, cv=5, scoring='neg_mean_absolute_error')
mae_model3 = -scores_model3.mean()
print("Model 3 Mean Absolute Error (CV):", round(mae_model3, 2))

#GridSearchCV for Model 1
param_grid = {
    'fit_intercept': [True, False]
}
grid_search = GridSearchCV(model1, param_grid={}, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(train_X, train_y)
best_params = grid_search.best_params_
print("Best Hyperparameters for Model 1:", best_params)

#GridSearchCV for Model 2
param_grid = {
    'n_estimators': [10, 30, 50],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
grid_search = GridSearchCV(model2, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(train_X, train_y)
best_params = grid_search.best_params_
print("Best Hyperparameters for Model 2:", best_params)
best_model2 = grid_search.best_estimator_

#GridSearchCV for Model 3
param_grid = {
    'criterion': ['friedman_mse', 'poisson', 'absolute_error', 'squared_error'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['log2', 'sqrt'],
    'max_leaf_nodes': [None, 10, 20, 30],
    'min_impurity_decrease': [0.0, 0.01, 0.1]
}
grid_search = GridSearchCV(model3, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(train_X, train_y)
best_params = grid_search.best_params_
print("Best Hyperparameters for Model 3:", best_params)
best_model3 = grid_search.best_estimator_

train_y = pd.DataFrame(train_y)
model1_prediction = pd.DataFrame(model1_prediction).astype(int)
model2_predictions = pd.DataFrame(model2_predictions).astype(int)
model3_predictions = pd.DataFrame(model3_predictions).astype(int)

#-----2.5 Model Performance Analysis-----

#Model 1
##Scores
accuracym1 = accuracy_score(train_y.iloc[:, 0], model1_prediction.iloc[:, 0])
precisionm1 = precision_score(train_y.iloc[:, 0], model1_prediction.iloc[:, 0], average= 'weighted')
f1m1 = f1_score(train_y.iloc[:, 0], model1_prediction.iloc[:, 0], average= 'weighted')
print("The accuracy score for Model 1 is:",round(accuracym1,2) )
print("The precision score for Model 1 is:",round(precisionm1,2) )
print("The F1 score for Model 1 is:",round(f1m1,2) )

##Confusion Matrix 
cm1 = confusion_matrix(train_y.iloc[:, 0], model1_prediction.iloc[:, 0])
class_labels = [str(i) for i in range(1, 14)]
sns.heatmap(cm1, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix for Model 1')
plt.xlabel('Predicted Step Value')
plt.ylabel('Actual Step Value')
plt.savefig("C:/Users/Owner/Documents/Python/GitHub/Plots/Confusion Matrix for Model 1")
plt.show()


#Model 2
##Scores
accuracym2 = accuracy_score(train_y.iloc[:, 0], model2_predictions.iloc[:, 0])
precisionm2 = precision_score(train_y.iloc[:, 0], model2_predictions.iloc[:, 0], average= 'weighted')
f1m2 = f1_score(train_y.iloc[:, 0], model2_predictions.iloc[:, 0], average= 'weighted')
print("The accuracy score for Model 2 is:",round(accuracym2,2))
print("The precision score for Model 2 is:",round(precisionm2,2))
print("The F1 score for Model 2 is:",round(f1m2,2))

##Confusion Matrix 
cm2 = confusion_matrix(train_y.iloc[:, 0], model2_predictions.iloc[:, 0])
class_labels = [str(i) for i in range(1, 14)]
sns.heatmap(cm2, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix for Model 2')
plt.xlabel('Predicted Step Value')
plt.ylabel('Actual Step Value')
plt.savefig("C:/Users/Owner/Documents/Python/GitHub/Plots/Confusion Matrix for Model 2")
plt.show()

#Model 3
##Scores
accuracym3 = accuracy_score(train_y.iloc[:, 0], model3_predictions.iloc[:, 0])
precisionm3 = precision_score(train_y.iloc[:, 0], model3_predictions.iloc[:, 0], average= 'weighted')
f1m3 = f1_score(train_y.iloc[:, 0], model3_predictions.iloc[:, 0], average= 'weighted')
print("The accuracy score for Model 3 is:",accuracym3 )
print("The precision score for Model 3 is:",precisionm3 )
print("The F1 score for Model 3 is:",f1m3 )

##Confusion Matrix 
cm3 = confusion_matrix(train_y.iloc[:, 0], model3_predictions.iloc[:, 0])
class_labels = [str(i) for i in range(1, 14)]
sns.heatmap(cm3, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix for Model 3')
plt.xlabel('Predicted Step Value')
plt.ylabel('Actual Step Value')
plt.savefig("C:/Users/Owner/Documents/Python/GitHub/Plots/Confusion Matrix for Model 3")
plt.show()

#-----2.6: Model Evaluation-----

joblib.dump(model3, 'Model3.joblib')
loaded_model3 = joblib.load('Model3.joblib')
validation_data = [[9.375,3.0625,1.51], [6.995,5.125,0.3875], [0,3.0625,1.93], [9.4,3,1.8], [9.4,3,1.3]]
validation_data_scaled = my_scaler.transform(validation_data)
predictions = loaded_model3.predict(validation_data_scaled)
print('The predicted values are: ', predictions)
