import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import missingno as msno
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import csv
import sys
import seaborn as sns


class XyScaler(BaseEstimator, TransformerMixin):
    """Standardize a training set of data along with a vector of targets.  """

    def __init__(self):
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def fit(self, X, y, *args, **kwargs):
        """Fit the scaler to data and a target vector."""
        self.X_scaler.fit(X)
        self.y_scaler.fit(y.reshape(-1, 1))
        return self

    def transform(self, X, y, *args, **kwargs):
        """Transform a new set of data and target vector."""
        return (self.X_scaler.transform(X),
                self.y_scaler.transform(y.reshape(-1, 1)).flatten())
def ResidualPlot(model):
    student_resid = model.outlier_test()['student_resid']
    plt.scatter(model.fittedvalues, student_resid)
    plt.xlabel('Fitted values of Target')
    plt.ylabel('Studentized Residuals')
    plt.savefig('ResidualPlot.png')
class er(BaseEstimator, TransformerMixin):
    """Standardize a training set of data along with a vector of targets.  """

    def __init__(self):
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def fit(self, X, y, *args, **kwargs):
        """Fit the scaler to data and a target vector."""
        self.X_scaler.fit(X)
        self.y_scaler.fit(y.reshape(-1, 1))
        return self

    def transform(self, X, y, *args, **kwargs):
        """Transform a new set of data and target vector."""
        return (self.X_scaler.transform(X),
                self.y_scaler.transform(y.reshape(-1, 1)).flatten())

def get_clean_data(file_name):
    df = pd.read_csv(file_name)
    print(df.head())
    df = df.drop(0)
    df['datetime']  =  pd.to_datetime(df['date'], format="%Y/%m/%d")
    df['volume'] = df['volume'].astype(float)
    df['days']  =  [float(days_between(x, df.date.min())) for x in df['date']]
    # df['Price1y']  =  [futureprice(df,x, 365) for x in df['days']]
    # df['Price4y']  =  [futureprice(df,x, 4*365) for x in df['days']]
    print(df.head())
    pd.plotting.scatter_matrix(df,figsize=(10, 8))
    plt.savefig("scatter_matrix.png")

    msno.matrix(df)
    plt.savefig('f3.png')
    temp = pd.DataFrame(df.isna().sum())

    temp.to_csv('nulls.csv', sep='\t', encoding='utf-8')
    #print(df.isna().sum())
    # one record had missing data multiple fields in one row ass

    # one record had missing data multiple fields in one row ass
    df = df.fillna(df.mean())
    msno.matrix(df)
    plt.savefig('f4.png')
    sns.pairplot(df1)
    plt.savefig('pairplot.png')
    return(df)



def rss(y, y_hat):
    return np.mean((y  - y_hat)**2)
def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y/%m/%d")
    d2 = datetime.strptime(d2, "%Y/%m/%d")
    return abs((d2 - d1).days)
from datetime import datetime



def cv(X, y, base_estimator, n_folds, random_seed=154):
    """Estimate the in and out-of-sample error of a model using cross validation.

    Parameters
    ----------

    X: np.array
      Matrix of predictors.

    y: np.array
      Target array.

    base_estimator: sklearn model object.
      The estimator to fit.  Must have fit and predict methods.

    n_folds: int
      The number of folds in the cross validation.

    random_seed: int
      A seed for the random number generator, for repeatability.

    Returns
    -------

    train_cv_errors, test_cv_errors: tuple of arrays
      The training and testing errors for each fold of cross validation.
    """
    kf = KFold(n_splits=n_folds, random_state=random_seed)
    test_cv_errors, train_cv_errors = np.empty(n_folds), np.empty(n_folds)
    for idx, (train, test) in enumerate(kf.split(X_train)):
        # Split into train and test
        X_cv_train, y_cv_train = X[train], y[train]
        X_cv_test, y_cv_test = X[test], y[test]
        # Standardize data.
        standardizer = XyScaler()
        standardizer.fit(X_cv_train, y_cv_train)
        X_cv_train_std, y_cv_train_std = standardizer.transform(X_cv_train, y_cv_train)
        X_cv_test_std, y_cv_test_std = standardizer.transform(X_cv_test, y_cv_test)
        # Fit estimator
        estimator = clone(base_estimator)
        estimator.fit(X_cv_train_std, y_cv_train_std)
        # Measure performance
        y_hat_train = estimator.predict(X_cv_train_std)
        y_hat_test = estimator.predict(X_cv_test_std)
        # Calclate the error metrics
        train_cv_errors[idx] = rss(y_cv_train_std, y_hat_train)
        test_cv_errors[idx] = rss(y_cv_test_std, y_hat_test)
    return train_cv_errors, test_cv_errors


def train_at_various_alphas(X, y, model, alphas, n_folds=10, **kwargs):
    """Train a regularized regression model using cross validation at various values of alpha.

    Parameters
    ----------

    X: np.array
      Matrix of predictors.

    y: np.array
      Target array.

    model: sklearn model class
      A class in sklearn that can be used to create a regularized regression object.  Options are `Ridge` and `Lasso`.

    alphas: numpy array
      An array of regularization parameters.

    n_folds: int
      Number of cross validation folds.

    Returns
    -------

    cv_errors_train, cv_errors_test: tuple of DataFrame
      DataFrames containing the training and testing errors for each value of
      alpha and each cross validation fold.  Each row represents a CV fold,
      and each column a value of alpha.
    """
    cv_errors_train = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),
                                     columns=alphas)
    cv_errors_test = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),
                                        columns=alphas)
    for alpha in alphas:
        train_fold_errors, test_fold_errors = cv(X, y, model(alpha=alpha, **kwargs), n_folds=n_folds)
        cv_errors_train.loc[:, alpha] = train_fold_errors
        cv_errors_test.loc[:, alpha] = test_fold_errors
    return cv_errors_train, cv_errors_test

def writeModelSummary(reportname, model):
    with open(reportname ,'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

def get_optimal_alpha(mean_cv_errors_test):
    alphas = mean_cv_errors_test.index
    optimal_idx = np.argmin(mean_cv_errors_test.values)
    optimal_alpha = alphas[optimal_idx]
    return optimal_alpha


if __name__ == '__main__':
    df = get_clean_data('data/HistoricalQuotes.csv')


    HIV_train, HIV_test = train_test_split(df, test_size=.3)
    HIV_train = HIV_train.copy()
    HIV_test = HIV_test.copy()

    target_column = "close"
    X_train, y_train = HIV_train.drop([target_column,'open','date','datetime'], axis=1), HIV_train[target_column]
    X_test, y_test = HIV_test.drop([target_column,'open','date','datetime'], axis=1), HIV_test[target_column]

    print(y_train.head())
    print(X_train.head())

    #sys.exit()
    ridge = Ridge(alpha=0.5)
    ridge.fit(X_train, y_train)
    #not working  writeModelSummary('ridge1.txt',ridge)
    preds = ridge.predict(X_test)
    mse = rss(y_test, preds)
    print("MSE for Ridge(alpha=0.5): {:2.2f}".format(mse))
    #not working  ResidualPlot(ridge)

    n_folds = 10
    train_cv_errors, test_cv_errors = cv(X_train.values, y_train.values,Ridge(alpha=0.5), n_folds=n_folds)


    print("Training CV error: {:2.2f}".format(train_cv_errors.mean()))
    print("Test CV error: {:2.2f}".format(test_cv_errors.mean()))

    ridge_alphas = np.logspace(-2, 4, num=250)

    ridge_cv_errors_train, ridge_cv_errors_test = train_at_various_alphas(
        X_train.values, y_train.values, Ridge, ridge_alphas)

    ridge_mean_cv_errors_train = ridge_cv_errors_train.mean(axis=0)
    ridge_mean_cv_errors_test = ridge_cv_errors_test.mean(axis=0)

    ridge_optimal_alpha = get_optimal_alpha(ridge_mean_cv_errors_test)

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(np.log10(ridge_alphas), ridge_mean_cv_errors_train)
    ax.plot(np.log10(ridge_alphas), ridge_mean_cv_errors_test)
    ax.axvline(np.log10(ridge_optimal_alpha), color='grey')
    ax.set_title("Ridge Regression Train and Test MSE")
    ax.set_xlabel(r"$\log(\alpha)$")
    ax.set_ylabel("MSE")
    fig.savefig('f2.png')

    ridge_models = []

    for alpha in ridge_alphas:
        scaler = XyScaler()
        scaler.fit(X_train.values, y_train.values)
        X_train_std, y_train_std = scaler.transform(X_train.values, y_train.values)
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_std, y_train_std)
        ridge_models.append(ridge)

    paths = pd.DataFrame(np.empty(shape=(len(ridge_alphas), len(X_train.columns))),
                         index=ridge_alphas, columns=X_train.columns)

    for idx, model in enumerate(ridge_models):
        paths.iloc[idx] = model.coef_

    fig, ax = plt.subplots(figsize=(14, 4))

    for column in X_train.columns:
        path = paths.loc[:, column]
        ax.plot(np.log10(ridge_alphas), path, label=column)
    ax.axvline(np.log10(ridge_optimal_alpha), color='grey')
    ax.legend(loc='lower right')
    ax.set_title("Ridge Regression, Standardized Coefficient Paths")
    ax.set_xlabel(r"$\log(\alpha)$")
    ax.set_ylabel("Standardized Coefficient")
    fig.savefig('f1.png')
