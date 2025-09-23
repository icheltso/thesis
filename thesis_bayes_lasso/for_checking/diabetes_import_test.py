# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 10:40:38 2025

@author: ichel
"""


from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
  
# fetch dataset 
diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296) 
  
# data (as pandas dataframes) 
X = diabetes_130_us_hospitals_for_years_1999_2008.data.features 
y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets 
  
# metadata 
print(diabetes_130_us_hospitals_for_years_1999_2008.metadata) 
  
# variable information 
print(diabetes_130_us_hospitals_for_years_1999_2008.variables) 


print(y['readmitted'].value_counts())
print(y['readmitted'].unique())
print(X.dtypes)

'''Drop weight as most values missing/corrupted'''
X = X.drop(columns=['weight']).copy()


'''Impute Nans with unknowns'''
X['max_glu_serum'].fillna('Unknown', inplace=True)
X['A1Cresult'].fillna('Unknown', inplace=True)
X['payer_code'].fillna('Unknown', inplace=True)
X['medical_specialty'].fillna('Unknown', inplace=True)
X['diag_1'].fillna('Unknown', inplace=True)
X['diag_2'].fillna('Unknown', inplace=True)
X['diag_3'].fillna('Unknown', inplace=True)
X['race'].fillna('Unknown', inplace=True)

# Step 4: Convert age into numeric values by extracting the lower bound of the ranges
age_mapping = {
    '[0-10)': 0,
    '[10-20)': 10,
    '[20-30)': 20,
    '[30-40)': 30,
    '[40-50)': 40,
    '[50-60)': 50,
    '[60-70)': 60,
    '[70-80)': 70,
    '[80-90)': 80,
    '[90-100)': 90
}
X['age'] = X['age'].map(age_mapping)


'''ordinally encode some ordered data'''
values_to_check = ['No', 'Up', 'Steady', 'Down']
value_mapping = {
    'No': 0,
    'Down': 1,
    'Steady': 2,
    'Up': 3
}
ordinal_columns = X.columns[X.applymap(lambda x: x in values_to_check).any()]
for col in ordinal_columns:
    X[col] = X[col].replace(value_mapping)

# Impute the rest of the missing values with the mode (for categorical features) or median (for numerical)
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

'''one-hot encode the other non-numeric data'''
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

'''Change to binary classification problem: 0 or 1 for no readmission/readmission'''
readmitted_mapping = {
    'NO': 0,     # No readmission
    '>30': 1,    # Readmission > 30 days
    '<30': 1     # Readmission < 30 days
}
y['readmitted'] = y['readmitted'].replace(readmitted_mapping)
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
model = LogisticRegression(penalty='l1', solver='saga', tol =1e-5)  # 'liblinear','saga' support L1

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print(f"Best C: {grid_search.best_params_['C']}")
print(f"Best Accuracy: {grid_search.best_score_}")

#Evaluate the best model from GridSearchCV on the test data
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Calculate the accuracy on the test set
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {test_accuracy:.4f}')


