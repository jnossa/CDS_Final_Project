import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split

def cross_validation(X: pd.DataFrame, y: pd.DataFrame, k: int=10):
    """
    Perform K-fold cross-validation for Linear Regression, Lasso, and Ridge models and compare MSE scores.

    Parameters:
    - X (pd.DataFrame): Features.
    - y (pd.DataFrame): Target variable.
    - k (int): Number of folds in the cross validation process. Default is 10.

    Returns:
    None (Prints MSE scores for each model.)
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

    # Initialize the models
    linear_model = LinearRegression()
    lasso_model = Lasso()
    ridge_model = Ridge()

    # Fit the models and obtain the mse
    linear_model.fit(X_train, y_train)
    lasso_model.fit(X_train, y_train)
    ridge_model.fit(X_train, y_train)
    linear_mse = np.mean((linear_model.predict(X_test) - y_test) ** 2)
    lasso_mse = np.mean((lasso_model.predict(X_test) - y_test) ** 2)
    ridge_mse = np.mean((ridge_model.predict(X_test) - y_test) ** 2)    

    # Use cross_val_score for cross-validated MSE
    linear_mse_cv_scores = -cross_val_score(linear_model, X, y, cv=k, scoring='neg_mean_squared_error')
    lasso_mse_cv_scores = -cross_val_score(lasso_model, X, y, cv=k, scoring='neg_mean_squared_error')
    ridge_mse_cv_scores = -cross_val_score(ridge_model, X, y, cv=k, scoring='neg_mean_squared_error')

    print("MSE for Linear model without cross-validation:", linear_mse)
    print("MSE for Lasso model with 10 fold cross validation: ", lasso_mse)
    print("MSE for Ridge model with 10 fold cross validation: ", ridge_mse)

    print("Cross-validated for Linear model MSE scores:", linear_mse_cv_scores)
    print("Mean MSE for Linear model with cross-validation:", np.mean(linear_mse_cv_scores))
    print("Cross-validated for Lasso model MSE scores:", lasso_mse_cv_scores)
    print("Mean MSE for Lasso model with cross-validation:", np.mean(lasso_mse_cv_scores))
    print("Cross-validated for Ridge model MSE scores:", ridge_mse_cv_scores)
    print("Mean MSE for Ridge model with cross-validation:", np.mean(ridge_mse_cv_scores))