# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 10:40:38 2025

@author: ichel
"""

import os
#os.environ['JAX_ENABLE_X64'] = 'True'
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np


"Import Higgs dataset and use 100k subsample to train lasso with logreg"


# fetch dataset 
# Load the compressed CSV
#df_all = pd.read_csv("HIGGS.csv.gz", compression="gzip", header=None)

#n = 100000  # Specify the desired sample size
#df = df_all.sample(n=n, random_state=42)  # Set random_state for reproducibility

# Set parameters
reload_var = True
#train_size = 10000
#test_size = 2000

subsample = 3000000  # Specify the desired sample size

if reload_var == True or not all(var in globals() for var in ["X", "y"]):
    
    if os.path.exists("higgs_data.npz") and reload_var == False:
        # Load from local file
        data = np.load("higgs_data.npz", allow_pickle = True)
        X, y = data["X"], data["y"]
        print("Forest dataset loaded from local file.")
    
    else:
        # Load the dataset
        df_all = pd.read_csv("HIGGS.csv.gz", compression="gzip", header=None)
        df = df_all.sample(n = subsample, random_state=42)  # Set random_state for reproducibility
        # data (as pandas dataframes) 
        X = df.iloc[:,1:]
        y = df.iloc[:,0]
        # Save dataset locally
        np.savez_compressed("higgs_data.npz", X=X, y=y)
        print("Higgs dataset loaded and subsample saved locally.")
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Finished train/test split")



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, C=0.1, tol =1e-5, max_iter=10000)
#model.fit(X_train_scaled, np.ravel(y_train))

#y_pred = model.predict(X_test_scaled)
#accuracy = accuracy_score(np.ravel(y_test), y_pred)

#print(f'Model Accuracy: {accuracy:.4f}')
print("Begin training")

#param_grid = {'C': [0.5, 0.55, 0.6]}
param_grid = {'C': [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1, 10]}
model = LogisticRegression(penalty='l1', solver='saga', tol =1e-5, max_iter = 10000)  # 'liblinear','saga' support L1

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose = 1)
grid_search.fit(X_train_scaled, y_train)

print(f"Best C: {grid_search.best_params_['C']}")
print(f"Best Accuracy: {grid_search.best_score_}")

#Evaluate the best model from GridSearchCV on the test data
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Calculate the accuracy on the test set
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {test_accuracy:.4f}')


