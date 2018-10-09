import numpy as np
import pandas as pd
import scipy as sc
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt

# 3D plots
from mpl_toolkits.mplot3d import Axes3D

# set larger font sizes and the style
import matplotlib as mpl
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, ax, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    p = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title,fontsize=font_size)

    plt.colorbar(p)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                 horizontalalignment="center", size = font_size,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    ax.set_ylabel('True label',fontsize=font_size)
    ax.set_xlabel('Predicted label',fontsize=font_size)
    fig.savefig('images/confusion.png')

font_size = 24
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams['xtick.labelsize'] = font_size-5
mpl.rcParams['ytick.labelsize'] = font_size-5
plt.style.use('bmh')

def plot_options(X,y):

    X_options = X[X['Options']==1 ]
    y_options = y[y['Options']==1 ]

    X_rsu = X[X['Options']==0 ]
    y_rsu  = y[y['Options']==0 ]

    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.scatter(X_rsu['mean_close'], X_rsu['ma_2'], color='b', label='RSU')
    ax.scatter(X_options['mean_close'], X_options['ma_2'], color='r', label='Options')
    ax.legend(shadow=True, fontsize='xx-large')
    ax.set_xlabel('mean_close',fontsize=font_size)
    ax.set_ylabel('Moving Average',fontsize=font_size)
    ax.set_title('Options or RSU?',fontsize=font_size)
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('data/clean_small2.csv')
    print(df.head())
    y_df = pd.DataFrame(df['Options'])
    y = df['Options']

    X = df[['mean_close','ma_2','mean_volume']]

    X['mean_log_vol'] = np.log(df['mean_volume'])
    X.drop('mean_volume', inplace=True, axis=1)
    #pd.plotting.scatter_matrix(X,figsize=(10, 8))
    #plt.savefig("images/scatter_matrix2.png")
    print(X.head())

    y_rsu  = y_df[y_df['Options']==0 ]
    print(y_rsu.head())
    #X = df.drop('Options',  axis=1)
    #plot_options(X,y)

    log_model = LogisticRegression()

    # LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
    #       intercept_scaling=1, max_iter=100, multi_class='warn',
    #       n_jobs=None, penalty='l2', random_state=None, solver='warn',
    #       tol=0.0001, verbose=0, warm_start=False)

    y_pred = log_model.fit(X,y).predict(X)
    print('y sum',sum(y))
    print('y_pred sum',sum(y_pred))
    y_true = y

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    print("| TN | FP |\n| FN | TP |\n")
    print(cnf_matrix)

    print(log_model.intercept_)
    print(log_model.coef_)

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.grid(False)
    class_names = ["RSU","Options"]
    plot_confusion_matrix(cnf_matrix, ax, classes=class_names,
                           title='Confusion matrix, without normalization')
