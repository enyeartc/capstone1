import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import missingno as msno
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
import csv
import sys
import seaborn as sns
from datetime import datetime
from datetime import timedelta

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant


daysinfuture = 365*4
thresholdVal = 0.45177236

def getMonday(a,b):
    return(a + timedelta(days=-b+1))
def getFutureMonday(a,b):
    date1 = a + timedelta(days=b)
    date1 = date1 + timedelta(days=(-date1.isoweekday()+1))
    return(date1)
def weeklyMean(date,df):
    df1 = df[df['MondayDate']==date]
    return(df1['close'].mean())
def weeklyVolume(date,df):
    df1 = df[df['MondayDate']==date]
    return(df1['volume'].mean())
def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y/%m/%d")
    d2 = datetime.strptime(d2, "%Y/%m/%d")
    return abs((d2 - d1).days)

def month_average(date,df):

    # Not working
    x = df.loc[df['MondayDate']== date].index[0]
    # x = df.index[df['MondayDate']== date]
    if x < 4:
        return(NaN)
    else:
        return((df.iloc[x]+df.iloc[x-1]+df.iloc[x-2]+df.iloc[x-3])/4)
def futureprice(df,currentday, numberofdaysinfuture):
    #return(df.loc[df['days'] == 50].close)
    if(df[df['days'] == currentday+numberofdaysinfuture].empty):
        return (0)
    else:
        return(df[df['days'] == currentday+numberofdaysinfuture]['close'].values[0])
def get_clean_data(file_name='data/OracleQuotes.csv', reset = 0):
    if reset != 1:
        df1 = pd.read_csv('data/clean_small2.csv')
    else:
        df = pd.read_csv(file_name)
        df = df.drop(0)
        df['datetime']  =  pd.to_datetime(df['date'], format="%Y/%m/%d")
        df.sort_values(['datetime'], inplace = True)
        df['volume'] = df['volume'].astype(float)
        df['days']  =  [float(days_between(x, df.date.min())) for x in df['date']]
        df['dayOfWeek'] = [x.isoweekday() for x in df['datetime']]
        df['MondayDate'] = [getMonday(a,b) for a,b in zip(df["datetime"], df["dayOfWeek"])]
        df['FutureMonday'] = [getFutureMonday(a,daysinfuture) for a in df["MondayDate"]]
        df['Price4y']  =  [futureprice(df,x, daysinfuture) for x in df['days']]
        df = df.replace(0, np.NaN)
        df['mean_close']  =  [weeklyMean(x,df) for x in df['MondayDate']]
        df['mean_volume']  =  [weeklyVolume(x,df) for x in df['MondayDate']]
        df['future_mean_close']  =  [weeklyMean(x,df) for x in df['FutureMonday']]
        df['future_mean_volume']  =  [weeklyVolume(x,df) for x in df['FutureMonday']]

        df.to_csv('data/clean.csv')

        # At this point we have created a dataframe with every day and every column_names
        # Now colapse and remove data with nulls
        msno.matrix(df)
        plt.savefig('images/msnoAllRows.png')
        #temp = pd.DataFrame(df.isna().sum())
        df1 = df[df['dayOfWeek']==1]

        df1['ma_1'] = df1['mean_close'].rolling(4).mean()
        df1['ma_2'] = df1['mean_close'].rolling(16).mean()
        #df['ma_2'] = df.rolling(4).mean()['close']

        df1.drop(['open', 'high', 'Price4y','low','days','dayOfWeek','date','datetime','volume','close'], inplace=True, axis=1)


        df1 = df1[df1['future_mean_close']>0 ]
        df1 = df1[df1['ma_2']>0 ]
        df1['fv_options'] = (df1['future_mean_close']-df1['mean_close'])*1000
        df1['fv_rsu'] = df1['future_mean_close']*250
        df1['diff'] = df1['fv_options']- df1['fv_rsu']

        # Adding 20 to insure that the value is always positive
        df1['ma2_mean'] = df1['ma_2']- df1['mean_close'] +20
        df1['ma1_mean'] = df1['ma_1']- df1['mean_close'] +20

        # All the work above was to git a value of 1 or 0 in this field
        df1['Options'] = [(lambda x: 1 if x >0 else 0)(x) for x in df1['diff']]

        print(df1.shape)
        df1.to_csv('data/clean_small2.csv')
        msno.matrix(df1)
        plt.savefig('images/msnoSubset.png')

    return(df1)

def make_clean_Options_df(df):

    # create the X parameters ZZZ
    #df = df[['Options','mean_close','ma_2','ma2_mean']]
    df = df[['Options','mean_close']]

    return (df)




def get_train_test(filepath):
    """Get standardized training and testing data sets from csv at given
    filepath.

    Parameters
    ----------
    filepath : str - path to find Options.csv

    Returns
    -------
    X_train : ndarray - 2D
    X_test  : ndarray - 2D
    y_train : ndarray - 1D
    y_test  : ndarray - 1D
    """
    raw_df = pd.read_csv(filepath)
    Options_df = make_clean_Options_df(raw_df)
    y = Options_df.pop('Options').values
    X = Options_df.values

    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,test_size=0.33)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def plotROC(y_true, y_prob_H, filelocation = 'images/plotROC.png',title = 'ROC curve'):
    #print("in plotRoc",filelocation,title)
    # print(y_true)
    # print(y_prob_H)
    # print('Length', len(y_prob_H))
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob_H, pos_label=1)
    print("****************",title, thresholds.max())
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
             label='Luck')
    ax.plot(fpr, tpr, color='b', lw=2, label='Model')
    ax.set_xlabel("False Positive Rate", fontsize=20)
    ax.set_ylabel("True Postive Rate", fontsize=20)
    ax.set_title(title, fontsize=24)
    ax.legend(fontsize=24);
    plt.savefig(filelocation)
def oversample0(X_train, y_train,count=10):
    # Quickly Oversample data with more RSU
    i = 0
    for Xt,yt in zip(X_train, y_train):
        if i < count:
            if(yt == 0):
                X_train = np.append(X_train,[Xt], axis=0)
                print('After',X_train)
                y_train = np.append(y_train,yt)
            i+=1
    return(X_train, y_train)

def get_train_test_data(df):
    raw_df = df
    Options_df = make_clean_Options_df(raw_df)
    y = Options_df.pop('Options').values
    X = Options_df.values

    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,test_size=0.33)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return(X_train, X_test, y_train, y_test)
    # After the split
    #X_train,y_train = oversample0(X_train,y_train)

def crossvalidation(X_train, X_test, y_train, y_test, thresh = thresholdVal):


    accuracies = []
    precisions = []
    recalls = []
    sumofy = []
    sumofy2 = []
    F1s = []
    kfold = StratifiedKFold(5,shuffle=False)

    for train_index, test_index in kfold.split(X_train,y_train):
        model = LR(solver='liblinear')
        model.fit(X_train[train_index], y_train[train_index])
        #y_predict = model.predict(X_train[test_index])
        y_proba = model.predict_proba(X_train[test_index])[:,1]
        #  Above the model is using predict_proba this returns the 'probability' of a 1
        #  the code below uses this with a differnt threshold to get the actual prediciton
        y_predict = np.array([(lambda z: 1 if z >thresh else 0)(z) for z in y_proba])
        y_true = y_train[test_index]
        #print(y_proba)
        #print('Predict',y_predict)
        #print('True   ',y_true)
        accuracies.append(accuracy_score(y_true, y_predict))
        precisions.append(precision_score(y_true, y_predict))
        recalls.append(recall_score(y_true, y_predict))
        sumofy.append(len(y_predict)-y_predict.sum())
        sumofy2.append(len(y_true)-y_true.sum())
        F1s.append(f1_score(y_true, y_predict))

    print ("---------Cross Validation---------")
    print ("Cross Validation accuracy:", np.average(accuracies))
    print ("Cross Validation precision:", np.average(precisions))
    print ("Cross Validation recall:", np.average(recalls))
    print ('Cross Validation F1', np.average(F1s))

def runFinalTests(X_train, X_test, y_train, y_test, thresh = thresholdVal):
    model = LR(solver='liblinear')
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_train)[:,1]
    y_predict = np.array([(lambda z: 1 if z >thresh else 0)(z) for z in y_proba])
    #print(y_train, y_predict)
    plotROC(y_train, y_proba,"images/plotROC_Training.png","ROC Curve Training")

    print ("---------Final Test---------")
    y_proba = model.predict_proba(X_test)[:,1]
    y_predict = np.array([(lambda z: 1 if z >thresh else 0)(z) for z in y_proba])
    print ('Test accuracy_score',accuracy_score(y_test, y_predict))
    print ('Test precision_score',precision_score(y_test, y_predict))
    print ('Test recall_score',recall_score(y_test, y_predict))
    print ('Test f1_score',f1_score(y_test, y_predict))

    print('------All ones')
    y_proba = model.predict_proba(X_train)[:,1]
    y_predict = np.array([(lambda z: 1 if z >0.0 else 0)(z) for z in y_proba])
    print ('All1 accuracy_score',accuracy_score(y_train, y_predict))
    print ('All1 precision_score',precision_score(y_train, y_predict))
    print ('All1 recall_score',recall_score(y_train, y_predict))
    print ('All1 f1_score',f1_score(y_train, y_predict))


def plot_options(filepath):
    df = pd.read_csv(filepath)
    X_options = df[df['Options']==1 ]
    y_options = df[df['Options']==1 ]

    X_rsu = df[df['Options']==0 ]
    y_rsu  = df[df['Options']==0 ]

    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.scatter(X_rsu['ma_2'], X_rsu['future_mean_close'], color='b', label='RSU')
    ax.scatter(X_options['ma_2'], X_options['future_mean_close'], color='r', label='Options')
    ax.legend(shadow=True)
    ax.set_xlabel('ma_2')
    ax.set_ylabel('Moving Average')
    ax.set_title('Options or RSU?')
    plt.savefig('images/OptionOrNot.png')
def find_threshold(X_train, X_test):

    kfold = StratifiedKFold(3,shuffle=False)

    accuracies = []
    precisions = []
    recalls = []
    sumofy = []
    sumofy2 = []
    F1s = []

    bestthr = 0
    bestF1  = 0
    #first get a list of threasholds, I will do this by first running a model
    # to get a list of potential threasholds run a quick model and get proba
    model = LR(solver='liblinear')
    model.fit(X_train, y_train)
    thr = model.predict_proba(X_train)[:,1]
    thr.sort()
    for i in thr:
        for train_index, test_index in kfold.split(X_train,y_train):
            model = LR(solver='liblinear')
            model.fit(X_train[train_index], y_train[train_index])
            #y_predict = model.predict(X_train[test_index])
            y_proba = model.predict_proba(X_train[test_index])[:,1]
            #  Above the model is using predict_proba this returns the 'probability' of a 1
            #  the code below uses this with a differnt threshold to get the actual prediciton
            y_predict = np.array([(lambda z: 1 if z >i else 0)(z) for z in y_proba])
            y_true = y_train[test_index]
            # with some thresholds there are no 1's predicted and an error
            # will be generated
            if(y_predict.sum() != 0):
                accuracies.append(accuracy_score(y_true, y_predict))
                precisions.append(precision_score(y_true, y_predict))
                recalls.append(recall_score(y_true, y_predict))
                sumofy.append(len(y_predict)-y_predict.sum())
                sumofy2.append(len(y_true)-y_true.sum())
                F1s.append(f1_score(y_true, y_predict))

                if(np.average(F1s) > bestF1):
                   bestF1 = np.average(F1s)
                   bestthr = i
            accuracies = []
            precisions = []
            recalls = []
            sumofy = []
            sumofy2 = []
            F1s = []
    #print('BestF1 and threashold',bestF1,bestthr)
    return(bestthr)
def makeplots(df):
    plt.figure(figsize=(16, 8))

def checkmodel(X_train,y_train):
    X = X_train
    X_const = add_constant(X, prepend=True)
    y = y_train
    logit_model = Logit(y, X_const).fit()
    print(logit_model.summary())


if __name__ == '__main__':
    np.random.seed(42)
    #threshold = thresholdVal
    # This does not need to be processed every time if the OracleQuotes
    # Data has not changed
    #df = get_clean_data('data/OracleQuotes.csv',1)
    df = get_clean_data()

    X_train, X_test, y_train, y_test = get_train_test_data(df)
    checkmodel(X_train,y_train)

    threshold = find_threshold(X_train, X_test)

    crossvalidation(X_train, X_test, y_train, y_test, threshold)
    runFinalTests(X_train, X_test, y_train, y_test, threshold)
