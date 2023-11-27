from sklearn.linear_model import LinearRegression, Lasso, Ridge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def find_best_alpha(X: pd.DataFrame, y: pd.DataFrame):
    """
    Find the best alpha and minimum MSE for Lasso and Ridge regression.

    Parameters:
    - X: Pandas DataFrame, features.
    - y: Pandas Series, target variable.

    Returns:
    - best_alpha_lasso: float, best alpha for Lasso.
    - min_mse_lasso: float, minimum MSE for Lasso.
    - best_alpha_ridge: float, best alpha for Ridge.
    - min_mse_ridge: float, minimum MSE for Ridge.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

    # Define a range of alpha values to try
    alphas = np.logspace(-10, 0, 50)

    # Initialize variables to store the best alpha and minimum MSE
    best_alpha_lasso = None
    best_alpha_ridge = None
    min_mse_lasso = float('inf')
    min_mse_ridge = float('inf')

    # Loop over the alpha values and fit Lasso and Ridge models
    for alpha in alphas:
        # Fit Lasso model
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train, y_train)
        y_pred_lasso = lasso.predict(X_test)
        mse_lasso = mean_squared_error(y_test, y_pred_lasso)

        # Update best alpha and minimum MSE for Lasso
        if mse_lasso < min_mse_lasso:
            min_mse_lasso = mse_lasso
            best_alpha_lasso = alpha

        # Fit Ridge model
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)
        mse_ridge = mean_squared_error(y_test, y_pred_ridge)

        # Update best alpha and minimum MSE for Ridge
        if mse_ridge < min_mse_ridge:
            min_mse_ridge = mse_ridge
            best_alpha_ridge = alpha

    return best_alpha_lasso, min_mse_lasso, best_alpha_ridge, min_mse_ridge

def find_best_regression_model(X: pd.DataFrame, y: pd.DataFrame):
    """
    Perform linear regression, Lasso regression, and Ridge regression with cross-validated alpha selection.

    Parameters:
    - X: Pandas DataFrame, features.
    - y: Pandas Series, target variable.

    Returns:
    - selected_model: sklearn model, the selected regression model.
    - selected_model_name: str, the name of the selected regression model.
    - formula: str, the regression formula for the best model.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)

    # Perform linear regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_mse = np.mean((linear_model.predict(X_test) - y_test) ** 2)

    # Perform Lasso regression to find the optimal alpha
    lasso_alpha, _, _, _ = find_best_alpha(X, y)
    lasso_model = Lasso(alpha=lasso_alpha)
    lasso_model.fit(X_train, y_train)
    lasso_mse = np.mean((lasso_model.predict(X_test) - y_test) ** 2)

    # Perform Ridge regression to find the optimal alpha
    ridge_alpha, _, _, _ = find_best_alpha(X, y)
    ridge_model = Ridge(alpha=ridge_alpha)
    ridge_model.fit(X_train, y_train)
    ridge_mse = np.mean((ridge_model.predict(X_test) - y_test) ** 2)

    print('MSE for Linear Regression:', round(linear_mse,2))
    print('MSE for LASSO:', round(lasso_mse,2))
    print('MSE for Ridge:', round(ridge_mse,2))

    # Choose the model with the lowest MSE
    if linear_mse <= lasso_mse and linear_mse <= ridge_mse:
        selected_model = linear_model
        selected_model_name = "Linear Regression"
    elif lasso_mse <= linear_mse and lasso_mse <= ridge_mse:
        selected_model = lasso_model
        selected_model_name = "Lasso Regression"
    else:
        selected_model = ridge_model
        selected_model_name = "Ridge Regression"

    # Print the regression formula for best model
    coef = selected_model.coef_
    intercept = selected_model.intercept_

    formula_parts = []
    for i, feature_name in enumerate(X.columns):
        coefficient = coef[i]
        if coefficient != 0:
            formula_parts.append(f"({coefficient:.2f} * {feature_name})")

    formula = f"{intercept:.2f} + {' + '.join(formula_parts)}"
    print(selected_model_name + " Regression Formula:")
    print(formula)

    # Optimal coefficients for each model
    coefs_linear = linear_model.coef_
    coefs_lasso = lasso_model.coef_
    coefs_ridge = ridge_model.coef_

    # Create bar plots for coefficients
    labels = X.columns

    # Create subplots with shared y-axis
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 15), sharey=True)

    # Set common y-axis limits
    y_min = min(min(coefs_linear), min(coefs_lasso), min(coefs_ridge))*1.2
    y_max = max(max(coefs_linear), max(coefs_lasso), max(coefs_ridge))*1.2

    # Linear Regression Coefficients
    bars_ridge = axes[0].bar(labels, coefs_linear, color='r', alpha=0.7)
    axes[0].set_title('Coefficients Linear')
    axes[0].set_xlabel('Features')
    axes[0].set_ylabel('Coefficient')
    axes[0].set_ylim([y_min, y_max])

    # Lasso Regression Coefficients
    bars_lasso = axes[1].bar(labels, coefs_lasso, color='b', alpha=0.7)
    axes[1].set_title('Coefficients Lasso')
    axes[1].set_xlabel('Features')
    axes[1].set_ylabel('Coefficient')
    axes[0].set_ylim([y_min, y_max])

    # Ridge Regression Coefficients
    bars_ridge = axes[2].bar(labels, coefs_ridge, color='g', alpha=0.7)
    axes[2].set_title('Coefficients Ridge')
    axes[2].set_xlabel('Features')
    axes[2].set_ylabel('Coefficient')
    axes[0].set_ylim([y_min, y_max])

    # Rotate x-axis labels
    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    # Annotate the values on the bars
    def annotate_bars(bars, ax):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    annotate_bars(bars_lasso, axes[0])
    annotate_bars(bars_ridge, axes[1])
    annotate_bars(bars_ridge, axes[2])

    plt.tight_layout()
    plt.show()

    return selected_model, selected_model_name, formula