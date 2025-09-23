# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 10:40:38 2025

@author: ichel
"""


from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Lasso
#from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes

# Load the dataset
#diabetes = load_diabetes()
#X = diabetes.data
#y = diabetes.target



# Load the dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
columns=diabetes.feature_names
        
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
alphas = [1e-3, 0.01, 0.1, 1, 10, 100]
param_grid = {'alpha': alphas}
#model = LogisticRegression(penalty='l1', solver='saga', tol =1e-5)  # 'liblinear','saga' support L1
model = Lasso()  # 'liblinear','saga' support L1

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

print(f"Best lambda: {grid_search.best_params_['alpha']}")
print(f"Best Accuracy: {grid_search.best_score_}")

#Evaluate the best model from GridSearchCV on the test data
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# Calculate the accuracy on the test set
test_mse = mean_squared_error(y_test, y_pred)
print(f'Test MSE: {test_mse:.4f}')

# Optionally, check the coefficients of the best model
best_lasso = grid_search.best_estimator_
# Get the coefficients from the best model
coefficients = best_lasso.coef_
# Normalize the coefficients by the standard deviation of the features
normalized_coeffs = coefficients / np.linalg.norm(coefficients)

print("Best model coefficients:", normalized_coeffs)


# Plot the normalized coefficients as a bar chart
plt.figure(figsize=(12, 6))  # Increase figure size for readability
plt.bar(columns, normalized_coeffs, color='blue')
plt.xticks(rotation=45, ha='right')  # Rotate feature names for better readability
plt.xlabel('Feature Name')
plt.ylabel('Normalized Coefficients')
plt.title('Optimal Coefficients from CV')
plt.tight_layout()  # Adjust the layout to prevent clipping of labels
plt.show()


# Plot bar charts of normalized coefficients for each alpha
plt.figure(figsize=(15, 10))
for i, alpha in enumerate(alphas, 1):
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)
    coefs = lasso.coef_
    norm_coefs = coefs / np.linalg.norm(coefs) if np.linalg.norm(coefs) != 0 else coefs

    plt.subplot(2, 3, i)
    plt.bar(columns, norm_coefs, color='blue')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Alpha = {alpha}')
    plt.ylim(-1.1, 1.1)  # same scale for all plots

# Get cross-validation MSE values for all alphas tested
results = pd.DataFrame(grid_search.cv_results_)
results = results[['param_alpha', 'mean_test_score']]
results['mean_test_score'] = -results['mean_test_score']  # Convert from negative MSE to positive MSE

# Print out the MSE values for each alpha
print("Mean Squared Errors for all alphas:")
print(results)

