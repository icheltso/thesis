# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 10:40:38 2025

@author: ichel
"""

import os
#os.environ['JAX_ENABLE_X64'] = 'True'
import sys
from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
  
# fetch dataset 
#covertype = fetch_ucirepo(id=31) 
  
# data (as pandas dataframes) 
#X = covertype.data.features 
#y = covertype.data.targets 
  
# metadata 
#print(covertype.metadata) 
  
# variable information 
#print(covertype.variables) 



# Set parameters
reload_var = False
#train_size = 10000
#test_size = 2000


if reload_var == True or not all(var in globals() for var in ["X_train", "X_test", "y_train", "y_test"]):
    
    if os.path.exists("forest_data.npz") and reload_var == False:
        # Load from local file
        data = np.load("forest_data.npz", allow_pickle = True)
        X, X_names, y = data["X"], data["X_names"], data["y"]
        print("Forest dataset loaded from local file.")
    
    else:
        # Load the dataset
        covertype = fetch_ucirepo(id=31) 
        # data (as pandas dataframes) 
        X = covertype.data.features 
        y = covertype.data.targets 
        # Save dataset locally
        np.savez_compressed("forest_data.npz", X=X, X_names = X.columns.astype(str), y=y)
        print("Forest dataset downloaded and saved locally.")


'''Modify labels so that type 2 trees are 1s and the rest 0s'''
y = (y == 2).astype(int)

'''one-hot encode the other non-numeric data'''
#X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42, C=0.1, tol =1e-5, max_iter=10000)
#model.fit(X_train_scaled, np.ravel(y_train))

#y_pred = model.predict(X_test_scaled)
#accuracy = accuracy_score(np.ravel(y_test), y_pred)

#print(f'Model Accuracy: {accuracy:.4f}')

#param_grid = {'C': [0.01, 0.1, 1]}
param_grid = {'C': [1e-5, 1e-4, 1e-3]}
model = LogisticRegression(penalty='l1', solver='saga', tol =1e-5, max_iter=10000, verbose=1)  # 'liblinear' supports L1

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, np.ravel(y_train))

print(f"Best C: {grid_search.best_params_['C']}")
print(f"Best Accuracy: {grid_search.best_score_}")

#Evaluate the best model from GridSearchCV on the test data
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Calculate the accuracy on the test set
test_accuracy = accuracy_score(np.ravel(y_test), y_pred)
print(f'Test Accuracy: {test_accuracy:.4f}')


