import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

# import data set
df = (pd.read_csv("Homework 3/forestfires.csv"))
# store the first 10 rows in variable
df_10 = df.head(10)
# print the 10 rows
print(df_10)

# prints the dataset size
print(df.size)

DMC_max = df['DMC'].max()
DMC_min = df['DMC'].min()
print("\nDMC min, max:")
print(DMC_min)
print(DMC_max)

DC_max = df['DC'].max()
DC_min = df['DC'].min()
print("\nDMC min, max:")
print(DC_min)
print(DC_max)

ISI_max = df['ISI'].max()
ISI_min = df['ISI'].min()
print("\nISI min, max:")
print(ISI_min)
print(ISI_max)

RH_max = df['RH'].max()
RH_min = df['RH'].min()
print("\nRH min, max:")
print(RH_min)
print(RH_max)

# wind
wind_max = df['wind'].max()
wind_min = df['wind'].min()
print("\nwind min, max:")
print(wind_min)
print(wind_max)

#rain
rain_max = df['rain'].max()
rain_min = df['rain'].min()
print("\nrain min, max:")
print(rain_min)
print(rain_max)

#size
RH_max = df['RH'].max()
RH_min = df['RH'].min()
print("\nRH min, max:")
print(RH_min)
print(RH_max)

# for the plot
# Task 2: Pie figure
pd.set_option('display.max_columns', 31)
y_count = df.size_category.value_counts().reset_index().rename(columns={'count': 'counts'})
plt.figure(figsize=(8,8))
plt.pie(y_count.counts, labels=y_count["size_category"], autopct='%1.2f%%', explode=(0,0.02))
# #plt.show()

# for the plot
# Task 2: Pie figure
pd.set_option('display.max_columns', 31)
y_count = df.monthaug.value_counts().reset_index().rename(columns={'count': 'counts'})
plt.figure(figsize=(8,8))
plt.pie(y_count.counts, labels=y_count["monthaug"], autopct='%1.2f%%', explode=(0,0.02))
plt.show()

month_df = df.groupby(['size_category', 'month']).size().reset_index().rename(columns={0: 'count'}).sort_values('count', ascending=False)
month_df.head(10)

plt.figure(figsize=(12,6))
sns.barplot(x="month", y= 'count', hue='size_category', data = month_df)
plt.title("Num of fires in each month", fontsize=17, y=1.02)
plt.xlabel('Month', fontsize=14)
plt.ylabel('Count', fontsize =14)
# #()

day_df = df.groupby(['size_category', 'day']).size().reset_index().rename(columns={0: 'count'}).sort_values('count', ascending=False)
day_df.head(10)

plt.figure(figsize=(7,6))
sns.barplot(x="day", y= 'count', hue='size_category', data = day_df)
plt.title("Num of fires in each day", fontsize=17, y=1.02)
plt.xlabel('Day', fontsize=14)
plt.ylabel('Count', fontsize =14)
# #()

# Quesiton 4 setup - size_category
labelencoder = LabelEncoder()
df.iloc[:,-1] = labelencoder.fit_transform(df.iloc[:,-1])
print(df['size_category'])
rain_df = df.groupby(['size_category', 'rain']).size().reset_index().rename(columns={0:'count'}).sort_values('count', ascending=False)
print(rain_df)

plt.figure(figsize=(12,6))
rain_df['size_category'] = rain_df['size_category'].astype(str)
sns.barplot(x='rain', y='count', hue='size_category', data=rain_df)
plt.title("Rainfall level in diff catefory of forest", y=1.02, fontsize=17)
plt.xlabel("Rain", fontsize=14)
plt.ylabel("count", fontsize=14)
#()
#  q4 RH
labelencoder = LabelEncoder()
df.iloc[:,-1] = labelencoder.fit_transform(df.iloc[:,-1])
print(df['RH'])
rain_df = df.groupby(['RH', 'rain']).size().reset_index().rename(columns={0:'count'}).sort_values('count', ascending=False)
print(rain_df)

plt.figure(figsize=(12,6))
rain_df['RH'] = rain_df['RH'].astype(str)
sns.barplot(x='rain', y='count', hue='RH', data=rain_df)
plt.title("Rainfall level in diff RH", y=1.02, fontsize=17)
plt.xlabel("Rain", fontsize=14)
plt.ylabel("count", fontsize=14)
#()

#  q4 FFMC
labelencoder = LabelEncoder()
df.iloc[:,-1] = labelencoder.fit_transform(df.iloc[:,-1])
print(df['FFMC'])
rain_df = df.groupby(['FFMC', 'rain']).size().reset_index().rename(columns={0:'count'}).sort_values('count', ascending=False)
print(rain_df)

plt.figure(figsize=(12,6))
rain_df['FFMC'] = rain_df['FFMC'].astype(str)
sns.barplot(x='rain', y='count', hue='FFMC', data=rain_df)
plt.title("Rainfall level in diff FFMC", y=1.02, fontsize=17)
plt.xlabel("Rain", fontsize=14)
plt.ylabel("count", fontsize=14)
#()

#  q4 DMC
labelencoder = LabelEncoder()
df.iloc[:,-1] = labelencoder.fit_transform(df.iloc[:,-1])
print(df['DMC'])
rain_df = df.groupby(['DMC', 'rain']).size().reset_index().rename(columns={0:'count'}).sort_values('count', ascending=False)
print(rain_df)

plt.figure(figsize=(12,6))
rain_df['DMC'] = rain_df['DMC'].astype(str)
sns.barplot(x='rain', y='count', hue='DMC', data=rain_df)
plt.title("Rainfall level in diff DMC", y=1.02, fontsize=17)
plt.xlabel("Rain", fontsize=14)
plt.ylabel("count", fontsize=14)
#()

#  q4 DC
labelencoder = LabelEncoder()
df.iloc[:,-1] = labelencoder.fit_transform(df.iloc[:,-1])
print(df['DC'])
rain_df = df.groupby(['DC', 'rain']).size().reset_index().rename(columns={0:'count'}).sort_values('count', ascending=False)
print(rain_df)

plt.figure(figsize=(12,6))
rain_df['DC'] = rain_df['DC'].astype(str)
sns.barplot(x='rain', y='count', hue='DC', data=rain_df)
plt.title("Rainfall level in diff DC", y=1.02, fontsize=17)
plt.xlabel("Rain", fontsize=14)
plt.ylabel("count", fontsize=14)
#()

#  q4 ISI
labelencoder = LabelEncoder()
df.iloc[:,-1] = labelencoder.fit_transform(df.iloc[:,-1])
print(df['ISI'])
rain_df = df.groupby(['ISI', 'rain']).size().reset_index().rename(columns={0:'count'}).sort_values('count', ascending=False)
print(rain_df)

plt.figure(figsize=(12,6))
rain_df['ISI'] = rain_df['ISI'].astype(str)
sns.barplot(x='rain', y='count', hue='ISI', data=rain_df)
plt.title("Rainfall level in diff ISI", y=1.02, fontsize=17)
plt.xlabel("Rain", fontsize=14)
plt.ylabel("count", fontsize=14)
#()

#  q4 temp
labelencoder = LabelEncoder()
df.iloc[:,-1] = labelencoder.fit_transform(df.iloc[:,-1])
print(df['temp'])
rain_df = df.groupby(['temp', 'rain']).size().reset_index().rename(columns={0:'count'}).sort_values('count', ascending=False)
print(rain_df)

plt.figure(figsize=(12,6))
rain_df['temp'] = rain_df['temp'].astype(str)
sns.barplot(x='rain', y='count', hue='temp', data=rain_df)
plt.title("Rainfall level in diff temp", y=1.02, fontsize=17)
plt.xlabel("Rain", fontsize=14)
plt.ylabel("count", fontsize=14)
#()

#  q4 wind
labelencoder = LabelEncoder()
df.iloc[:,-1] = labelencoder.fit_transform(df.iloc[:,-1])
print(df['wind'])
rain_df = df.groupby(['wind', 'rain']).size().reset_index().rename(columns={0:'count'}).sort_values('count', ascending=False)
print(rain_df)

plt.figure(figsize=(12,6))
rain_df['wind'] = rain_df['wind'].astype(str)
sns.barplot(x='rain', y='count', hue='wind', data=rain_df)
plt.title("Rainfall level in diff wind", y=1.02, fontsize=17)
plt.xlabel("Rain", fontsize=14)
plt.ylabel("count", fontsize=14)
#()

#  q4 area
labelencoder = LabelEncoder()
df.iloc[:,-1] = labelencoder.fit_transform(df.iloc[:,-1])
print(df['area'])
rain_df = df.groupby(['area', 'rain']).size().reset_index().rename(columns={0:'count'}).sort_values('count', ascending=False)
print(rain_df)

plt.figure(figsize=(12,6))
rain_df['area'] = rain_df['area'].astype(str)
sns.barplot(x='rain', y='count', hue='area', data=rain_df)
plt.title("Rainfall level in diff area", y=1.02, fontsize=17)
plt.xlabel("Rain", fontsize=14)
plt.ylabel("count", fontsize=14)
#()

# question 5
# drop unnessary cols
df.drop(['month', 'day', 'monthjan', 'daymon'], axis=1, inplace=True)
pd.set_option("display.max_columns", 27)
df.head()
# check the outliers
from sklearn.ensemble import IsolationForest
data1 = df.copy()

# train the model
clf= IsolationForest(random_state=10, contamination=.01)
clf.fit(data1)
data1['anomoly'] = clf.predict(data1.iloc[:,0:27])
outliers = data1[data1['anomoly']==-1]
print('OUTLIERS:')
print(outliers)

# Remove outliers
outliers.index
df.drop([281, 299, 379, 463, 464, 469], axis=0, inplace=True)
print('\nSHAPE:')
print(df.shape)

# split data into target vars and independent vars
x = df.drop('size_category', axis = 1)
y = df['size_category']

# converting ind feat. into norm and standard data
norm = MinMaxScaler()
std = StandardScaler()

x_norm = pd.DataFrame(norm.fit_transform(x), columns=x.columns)
x_std = pd.DataFrame(std.fit_transform(x), columns=x.columns)
print('\nNORMAL AND STANDARD:')
print(x_std.head())

# Creating train and test data for model validation, with the train-test-rate = 3:1 (0.75:0.25)
# norm
x_train_norm, x_test_norm, y_train_norm, y_test_norm = train_test_split(x_norm, y, test_size=0.25)
# standard
x_train_std, x_test_std, y_train_std, y_test_std = train_test_split(x_std, y, test_size=0.25)

print('\nNORMAL:')
print('x_train_norm: ', x_train_norm.shape)
print('x_test_norm: ', x_test_norm.shape)
print('y_train_norm: ', y_train_norm.shape)
print('y_test_norm: ', y_test_norm.shape)

print('\nSTANDARD:')
print('x_train_std: ', x_train_std.shape)
print('x_test_std: ', x_test_std.shape)
print('y_train_std: ', y_train_std.shape)
print('y_test_std: ', y_test_std.shape)

# standardized data:
# y_train_std = y_train_std.astype('int')
# y_test_std = y_test_std.astype('int')
# clf = SVC()
# param_grids = [{'kernel':['rfb'], 'C': [15,14,13,12,11,10,1,0.1,0,0.001]}]
# grid = GridSearchCV(clf, param_grids, cv=20)
# grid.fit(x_train_std, y_train_std)
# print(grid.best_score_,grid.best_params_)
# kernel = rfb
x_train = x_train_norm.astype('int')
x_test = x_test_norm.astype('int')
y_train=y_train_norm.astype('int')
y_test=y_test_norm.astype('int')
clf = SVC()
param_grids = [{'kernel':['rbf'], 'C': [15,14,13,12,11,10,1,0.1,0.001]}]
grid = GridSearchCV(clf, param_grids, cv=20)
grid. fit(x_train,y_train)
print("\nBest C value")
print(grid.best_score_)
print("\nBest grid params")
print(grid.best_params_)

# use best C value from grid search (in output)
best_C = 0.7518421052631579 

# use scale and auto for gamma vals
gamma_values = ['scale', 'auto']

for gamma_val in gamma_values:
    best_svc = SVC(kernel='rbf', C=best_C, gamma=gamma_val)

    # train model
    best_svc.fit(x_train, y_train)

    # predictions
    y_train_pred = best_svc.predict(x_train_norm)
    y_test_pred = best_svc.predict(x_test_norm)

    # calculate accuracy for training and testing
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # print results
    print(f'\nRBF kernel with gamma={gamma_val}:')
    print('Train Accuracy:', train_accuracy)
    print('Test Accuracy:', test_accuracy)
    
    # Define the parameter grids for grid search
param_grids_poly = [{'kernel': ['poly'], 'C': [15, 14, 13, 12, 11, 10, 1, 0.1, 0.001], 'degree': [2, 3, 4]}]
param_grids_linear = [{'kernel': ['linear'], 'C': [15, 14, 13, 12, 11, 10, 1, 0.1, 0.001]}]

# grid search for polynomial
grid_poly = GridSearchCV(SVC(), param_grids_poly, cv=5)
grid_poly.fit(x_train, y_train)

# grid search for linear
grid_linear = GridSearchCV(SVC(), param_grids_linear, cv=5)
grid_linear.fit(x_train, y_train)

# set the best parameters for poly and linear
best_params_poly = grid_poly.best_params_
best_params_linear = grid_linear.best_params_

# use best parameters in SVC instance for poly and linear
best_svc_poly = SVC(**best_params_poly)
best_svc_linear = SVC(**best_params_linear)

# train using x_train_norm and x_train_standard
best_svc_poly.fit(x_train, y_train)
best_svc_linear.fit(x_train, y_train)

# predictions
y_train_pred_poly = best_svc_poly.predict(x_train)
y_test_pred_poly = best_svc_poly.predict(x_test)
y_train_pred_linear = best_svc_linear.predict(x_train)
y_test_pred_linear = best_svc_linear.predict(x_test)

# calculate accuracy for training and testing data
train_accuracy_poly = accuracy_score(y_train, y_train_pred_poly)
test_accuracy_poly = accuracy_score(y_test, y_test_pred_poly)
train_accuracy_linear = accuracy_score(y_train, y_train_pred_linear)
test_accuracy_linear = accuracy_score(y_test, y_test_pred_linear)

# output the results
print('\nPolynomial Kernel:')
print('Train Accuracy:', train_accuracy_poly)
print('Test Accuracy:', test_accuracy_poly)

print('\nLinear Kernel:')
print('Train Accuracy:', train_accuracy_linear)
print('Test Accuracy:', test_accuracy_linear)



# Q7 Problem statement 2?? salaray data????