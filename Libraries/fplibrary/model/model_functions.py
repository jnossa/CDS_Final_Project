import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold

class Model:
    """
    Model class for training and predicting using a machine learning model.

    Attributes:
        _features (list): List of feature columns used for training and prediction.
        _target (str): Target column used for training and prediction.
        model: Machine learning model for training and prediction.

    Methods:
        __init__(self, features, target, model, hyperparameters=None)
            Initializes the Model class with the specified parameters.

        train(self, df_train)
            Trains the model using the provided training data.

        predict(self, data)
            Predicts using the trained model on the provided data and returns predicted probabilities.

        accuracy(self, data, pred_col)
            Reports accuracy of predictions.
    """
    def __init__(self, features, target, model, hyperparameters=None):
        self._features = features
        self._target = target
        self.model = model(**hyperparameters) if hyperparameters else model

    def train(self, df_train):
        """
        Trains the model using the provided training data.

        Parameters:
        - df_train (pd.DataFrame): DataFrame containing the training data.
        """
        self.model.fit(df_train[self._features], df_train[self._target])
    
    def predict(self, data):
        """
        Predicts using the trained model on the provided data and returns predicted probabilities.

        Parameters:
        - data (pd.DataFrame): DataFrame containing the data to make predictions on.

        Returns:
        - np.array: Predicted values.
        """
        X_test = data[self._features]

        return self.model.predict(X_test)
    
    def accuracy(self, data, pred):
        """
        Reports the mean squared error accuracy of predictions.

        Parameters:
        - data (pd.DataFrame): DataFrame containing the true target values.
        - pred (np.array): Predicted values.

        Returns:
        - float: Mean squared error accuracy.
        """
        return mean_squared_error(data[self._target], pred)

class CrossValidation:
    """
    A class for performing cross-validation and reporting accuracy metrics.

    Methods:
    - regression_accuracy(X, y, model)
      Perform 5-fold cross-validation for regression and print mean squared error.

    - classification_accuracy(X, y, model)
      Perform 10-fold cross-validation for classification and print accuracy.

    Attributes:
    - None
    """
    def __init__(self, features, target, model, hyperparameters=None):
        self._features = features
        self._target = target
        self.model = model(**hyperparameters) if hyperparameters else model
        
    # Performing 5-fold cross-validation
    def regression_accuracy(self, data):
        """
        Perform 5-fold cross-validation for regression and print mean squared error.

        Parameters:
        - data (pd.DataFrame): DataFrame containing the data.

        Example:
        ```python
        # Perform regression cross-validation on the data
        cross_validator.regression_accuracy(data)
        ```
        """
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_val_score(self.model, data[self._features], data[self._target], cv=kf, scoring='neg_mean_squared_error')

        # Printing the cross-validation results
        print("Cross-validation results:")
        print(cv_results*(-1))
        print(f"Mean squared error: {(cv_results.mean())*-1:.2f} +/- {cv_results.std():.2f}")
    
    def classification_accuracy(self, data):
        """
        Perform 10-fold cross-validation for classification and print accuracy.

        Parameters:
        - data (pd.DataFrame): DataFrame containing the data.

        Example:
        ```python
        # Perform classification cross-validation on the data
        cross_validator.classification_accuracy(data)
        ```
        """
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_results = cross_val_score(self.model, data[self._features], data[self._target], cv=kf)

        # Printing the cross-validation results
        print("Cross-validation results:")
        print(cv_results)
        print(f"Accuracy: {cv_results.mean()} +/- {cv_results.std():.3f}")
        

class HyperparameterTuning:
    """
    A class for performing hyperparameter tuning for Lasso and Ridge regression models.

    Methods:
        find_best_alpha(data)
            Find the best alpha and minimum Mean Squared Error (MSE) for Lasso and Ridge regression.

    Attributes:
        _features (list): List of feature columns used for regression.
        _target (str): Name of the target variable for regression.
    """
    def __init__(self, features, target):
        self._features = features
        self._target = target
        

    def find_best_alpha(self, data):
        """
        Find the best alpha and minimum MSE for Lasso and Ridge regression.

        Parameters:
        - data: data to run the hyperparameter tuning with.

        Returns:
        - best_alpha_lasso: float, best alpha for Lasso.
        - min_mse_lasso: float, minimum MSE for Lasso.
        - best_alpha_ridge: float, best alpha for Ridge.
        - min_mse_ridge: float, minimum MSE for Ridge.
        """
        # Split the data into training and testing sets
        X, X_test, y, y_test = train_test_split(data[self._features], data[self._target], test_size=0.2, random_state=28)

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
            lasso.fit(X, y)
            y_pred_lasso = lasso.predict(X_test)
            mse_lasso = mean_squared_error(y_test, y_pred_lasso)

            # Update best alpha and minimum MSE for Lasso
            if mse_lasso < min_mse_lasso:
                min_mse_lasso = mse_lasso
                best_alpha_lasso = alpha

            # Fit Ridge model
            ridge = Ridge(alpha=alpha)
            ridge.fit(X, y)
            y_pred_ridge = ridge.predict(X_test)
            mse_ridge = mean_squared_error(y_test, y_pred_ridge)

            # Update best alpha and minimum MSE for Ridge
            if mse_ridge < min_mse_ridge:
                min_mse_ridge = mse_ridge
                best_alpha_ridge = alpha

        return best_alpha_lasso, min_mse_lasso, best_alpha_ridge, min_mse_ridge
        


class ModelCoefficients:
    """
    A class for obtaining and visualizing coefficients of linear regression, Lasso regression, and Ridge regression models.

    Methods:
    - get_coefficients(X, y)
      Get the intercept and coefficients of linear, Lasso, and Ridge regression models.

    - plot_coefficients(X, y)
      Plot bar plots for the coefficients of linear, Lasso, and Ridge regression models.

    Attributes:
    - None
    """
    def __init__(self) -> None:
        pass

    def get_coefficients(self, linear_model, lasso_model, ridge_model, X):
        """
        Get the intercept and coefficients of linear, Lasso, and Ridge regression models.

        Parameters:
        - linear_model: sklearn.linear_model.LinearRegression
        The trained linear regression model.
        
        - lasso_model: sklearn.linear_model.Lasso
        The trained Lasso regression model.

        - ridge_model: sklearn.linear_model.Ridge
        The trained Ridge regression model.

        Returns:
        - linear_intercept: float
        The intercept of the linear regression model.
        
        - linear_coef: np.ndarray
        Coefficients of the linear regression model.
        
        - lasso_intercept: float
        The intercept of the Lasso regression model.
        
        - lasso_coef: np.ndarray
        Coefficients of the Lasso regression model.
        
        - ridge_intercept: float
        The intercept of the Ridge regression model.
        
        - ridge_coef: np.ndarray
        Coefficients of the Ridge regression model.
        """

        # Print the regression formulas
        linear_coef = linear_model.coef_
        linear_intercept = linear_model.intercept_
        lasso_coef = lasso_model.coef_
        lasso_intercept = lasso_model.intercept_
        ridge_coef = ridge_model.coef_
        ridge_intercept = ridge_model.intercept_

        linear_formula_parts = []
        for i, feature_name in enumerate(X.columns):
            coefficient = linear_coef[i]
            if coefficient != 0:
                linear_formula_parts.append(f"({coefficient:.2f} * {feature_name})")

        linear_formula = f"{linear_intercept:.2f} + {' + '.join(linear_formula_parts)}"
        print('Linear Model' + " Regression Formula:")
        print(linear_formula)

        lasso_formula_parts = []
        for i, feature_name in enumerate(X.columns):
            coefficient = lasso_coef[i]
            if coefficient != 0:
                lasso_formula_parts.append(f"({coefficient:.2f} * {feature_name})")

        lasso_formula = f"{lasso_intercept:.2f} + {' + '.join(lasso_formula_parts)}"
        print('Lasso Model' + " Regression Formula:")
        print(lasso_formula)

        ridge_formula_parts = []
        for i, feature_name in enumerate(X.columns):
            coefficient = ridge_coef[i]
            if coefficient != 0:
                ridge_formula_parts.append(f"({coefficient:.2f} * {feature_name})")

        ridge_formula = f"{ridge_intercept:.2f} + {' + '.join(ridge_formula_parts)}"
        print('Ridge Model' + " Regression Formula:")
        print(ridge_formula)

        return linear_intercept, linear_coef, lasso_intercept, lasso_coef, ridge_intercept, ridge_coef
    
    def plot_coefficients(self, linear_model, lasso_model, ridge_model, X):
        """
        Plot bar plots for the coefficients of linear, Lasso, and Ridge regression models.

        Parameters:
        - X: Pandas DataFrame, features.
        - y: Pandas Series, target variable.

        Returns:
        None
        """
        # Get coefficients
        _, linear_coef, _, lasso_coef, _, ridge_coef = self.get_coefficients(linear_model, lasso_model, ridge_model, X)

        # Create bar plots for coefficients
        labels = X.columns

        # Create subplots with shared y-axis
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 15), sharey=True)

        # Set common y-axis limits
        y_min = min(min(linear_coef), min(lasso_coef), min(ridge_coef))*1.2
        y_max = max(max(linear_coef), max(lasso_coef), max(ridge_coef))*1.2

        # Linear Regression Coefficients
        bars_ridge = axes[0].bar(labels, linear_coef, color='r', alpha=0.7)
        axes[0].set_title('Coefficients Linear')
        axes[0].set_xlabel('Features')
        axes[0].set_ylabel('Coefficient')
        axes[0].set_ylim([y_min, y_max])

        # Lasso Regression Coefficients
        bars_lasso = axes[1].bar(labels, lasso_coef, color='b', alpha=0.7)
        axes[1].set_title('Coefficients Lasso')
        axes[1].set_xlabel('Features')
        axes[1].set_ylabel('Coefficient')
        axes[0].set_ylim([y_min, y_max])

        # Ridge Regression Coefficients
        bars_ridge = axes[2].bar(labels, ridge_coef, color='g', alpha=0.7)
        axes[2].set_title('Coefficients Ridge')
        axes[2].set_xlabel('Features')
        axes[2].set_ylabel('Coefficient')
        axes[0].set_ylim([y_min, y_max])

        # Rotate x-axis labels
        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')


        plt.tight_layout()
        plt.show()
