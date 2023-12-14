# Predictive Analytics Library

## Overview and Objective

The objective of this project is to build the architecture of a scalable and functional library focused on predictive analytics. The project was developed as the final project for a Computing for Data Science course  within the Data Science for Decision Making Master's program at the Barcelona School of Economics.

The library is designed to handle tasks such as loading data, preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation. The goal is to create a structured and extensible framework that can be easily scaled as the project evolves.

## Collaborators

* Viktoriia Yuzkiv
* Angelo Di Gianvito 
* Oliver Gatland
* Joaquin Ossa

## Dataset

The dataset used in this project contains apartment offers from the 15 largest cities in Poland (Warsaw, Lodz, Krakow, Wroclaw, Poznan, Gdansk, Szczecin, Bydgoszcz, Lublin, Katowice, Bialystok, Czestochowa). The data is sourced from local websites with apartments for sale and is supplemented with data from Open Street Map, providing distances to points of interest (POI). The dataset covers the timespan between August 2023 and November 2023. You can find the dataset [here](https://www.kaggle.com/datasets/krzysztofjamroz/apartment-prices-in-poland).

## Repository Structure

The project repository is organized with the following structure:
```
- /Library                          # Folder for the library
    - fplibrary                     # Source code for the library
        - /loading_data             # Modules to load and split dataset
        - /exploration              # Modules to create plots to explore the data
        - /preprocessing            # Modules for handling missing values and outliers
        - /feature_engineering      # Modules for feature engineering
        - /models                   # Modules for machine learning models
    - LICENCE                       # Library licence
    - setup.py                      # Setup file for the library
    - requirements.txt              # Dependencies for the library
- /data                             # Folder for dataset files
- /test                             # Unit tests for the library
- /notebook.ipynb                   # Jupyter Notebook demonstrating pipepline for library usage
- README.md                         # Project overview and guidelines
```

## Library Usage

To use the library, install the library on your local machine. In order to do so, run the following command while in the Library folder:
```
pip install -e .
```
- Import the necessary functions and classes from the library.
- Load the dataset and create an instance of the library classes.
- Build an end-to-end pipeline using the provided functions and classes.
- Run the pipeline to load data, preprocess it, create features, train a model, perform hyperparameter tuning, and evaluate predictions.

## Example Pipeline

The end-to-end pipeline is provided in the file `notebook.ipynb`.

Example of the pipeline that can be created:

```
# Import necessary modules from the library
import fplibrary.exploration as e
import fplibrary.feature_engineering as f
import fplibrary.loading_data as l
import fplibrary.preprocessing as p
import fplibrary.model as m

# Load data
data_loader = l.DataLoader('data.csv')
train_data, test_data = data_loader.load_and_split_data()

# Example of data exploration
plots = e.ExplorationPlots(all_data)
plots.create_boxplot('target_variable', ['feature_1', 'feature_2'], n_rows=1, n_cols=2)

# Example of data processing
miss_fill = p.MissingValues(all_data)
all_data = miss_fill.impute_missing_knn(5)

# Example of feature engineering
weather = f.WeatherFeatures(all_data)
all_data = weather.obtain_weather()

# Example of modelling
lin_model = LinearRegression()
lin_model_runner = m.Model(features, 'price', lin_model)
lin_model_runner.train(train_data)
lin_predict = lin_model_runner.predict(test_data)
lin_model_runner.accuracy(test_data, lin_predict)
```

## Unit Tests

The library includes comprehensive unit tests to ensure the correctness of its functionalities. You can run the tests using the following command:
 ```
 pytest test/
 ```

##  Guidelines for Scaling the Library

Below are the guidelines to scale the library and facilitate the addition of new preprocessors, features, and models:

**1. Modular Folder Structure:**

Follow the existing modular folders structure that is based on functionality. E.g., new preprocessors should be added to the `preprocessing` folder, new features - to `feature_engineering` respectively. 

**2. Consistent Naming Conventions:**

Maintain a consistent naming convention for files, classes, and functions across the library.
Use meaningful names that reflect the purpose of the component.

**3. Documentation:**

Provide detailed documentation for each module, class, and function.
Include information on input parameters, expected formats, and return values.
Clearly outline the purpose and usage of each component.

**4. Use of Classes and Inheritance:**

Design preprocessors, features, and models as classes, allowing for easy extension and customization.
Utilize inheritance to create a hierarchy of classes, enabling the reuse of common functionalities.

**5. Configurability:**

Allow users to easily adjust settings and parameters based on their specific requirements.

**6. Version Control:**

Clearly document release notes, specifying any breaking changes or new features added.

**7. Unit Testing:**

Develop comprehensive unit tests for each component.
Ensure that tests cover a range of scenarios and edge cases to guarantee robustness.

**8. Example Notebooks:**

Include Jupyter notebooks or example scripts demonstrating how to use the library for end-to-end tasks.
Showcase how to integrate new preprocessors, features, or models in these examples.

##  How to contribute
We welcome contributions from the community to enhance and improve this project. If you are interested in contributing, there are various areas where your expertise can make a significant impact:

### Feature Enhancements
**- New Features:** Propose and implement new features that align with the project's goals. Check the existing issues or suggest your own ideas.

**- Model Evaluation Metrics:** Contribute new model evaluation metrics or improve existing ones to enhance the library's capabilities.

**- Data Processing Methods:** Introduce new methods for data processing or improve existing ones to make the library more versatile.

### Bug Fixes
**- Bug Reports:** Help us identify and fix bugs by reporting them through GitHub issues. Include detailed descriptions and steps to reproduce.

**- Bug Fixes:** Address existing bugs by providing solutions through pull requests.

### Documentation
**- Documentation Improvements:** Enhance the clarity and completeness of our documentation. Updates to guides, tutorials, and examples are highly valuable.

**- Code Comments:** Improve code comments to make the codebase more understandable for contributors and users.

### Code Quality
**- Code Reviews:** Participate in code reviews to maintain code quality and consistency. Offer constructive feedback and suggestions.

**- Coding Standards:** Ensure that code contributions adhere to the established coding standards.
