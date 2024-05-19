# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd

# Custom imports
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, r2_score
import copy
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training and test data
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # Dummy initialization of the X_train, X_test and y_train
    # TODO: Depending on how you deal with the non-numeric data, you may want to 
    # modify/ignore the initialization of these variables
    #X_train = np.zeros_like(train_df.drop(['price_CHF'],axis=1))
    #y_train = np.zeros_like(train_df['price_CHF'])
    #X_test = np.zeros_like(test_df)
    
    # Preprocess 
    train_df_preprocessed = copy.deepcopy(train_df)
    test_df_preprocessed = copy.deepcopy(test_df)
    
    train_interpolated = train_df_preprocessed.interpolate(method='linear', limit_direction='both')
    test_interpolated = test_df_preprocessed.interpolate(method='linear', limit_direction='both')
    
    X_train = train_interpolated.drop(['price_CHF'], axis=1)
    y_train = train_interpolated['price_CHF']
    X_test = test_interpolated
    
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(), ['season'])],
        remainder='passthrough')
    
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    
    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """
    #y_pred=np.zeros(X_test.shape[0])
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions
    kernels = [RBF(), DotProduct(), Matern(), RationalQuadratic()]
    tscv = TimeSeriesSplit(n_splits=5)
    regressor = GaussianProcessRegressor(random_state=42, n_restarts_optimizer=0)
    param_grid = {'kernel': kernels}

    grid_search = GridSearchCV(regressor, param_grid, cv=tscv, scoring=make_scorer(r2_score))
    grid_search.fit(X_train, y_train)
    
    print(f"Best params: {grid_search.best_params_}")
    best_regressor = grid_search.best_estimator_
    
    y_pred = best_regressor.predict(X_test)

    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

