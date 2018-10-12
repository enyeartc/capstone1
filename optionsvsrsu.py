import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone,BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
import csv
from datetime import datetime, timedelta
from getstockdata import get_clean_data
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant


daysinfuture = 365*4
thresholdVal = 0.45177236

def make_clean_Options_df(df):
    ''' This funciton will define the X values
        it started out with many differnt values at the end of the day
        it ended up with just the mean
    '''
    df = df[['Options','ma2_mean']]

    return (df)

def plotROC(y_true, y_prob_H, filelocation = 'images/plotROC.png',title = 'ROC curve'):
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
    # In the case where I need to add more RSU data (the minority)
    # this will add rows with the training data
    i = 0
    for Xt,yt in zip(X_train, y_train):
        if i < count:
            if(yt == 0):
                X_train = np.append(X_train,[Xt], axis=0)
                print('After',X_train)
                y_train = np.append(y_train,yt)
            i+=1
    return(X_train, y_train)

def get_train_test_data(raw_df):
    ''' Take the dataframe passed in
        and pop out the Target (y)
        split the data and return the values
    '''
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

    return(model)

def plot_options(df, model,t):
    #df = pd.read_csv(filepath)

    Options_df = make_clean_Options_df(df)
    y = Options_df.pop('Options').values
    X = Options_df.values
    model.fit(X, y)
    y_proba = model.predict_proba(X)[:,1]
    y_predict = np.array([(lambda z: 1 if z >t else 0)(z) for z in y_proba])
    df['modelOptions'] = y_predict

    X_modelOptions = df[df['modelOptions']==1 ]
    y_modelOptions = df[df['modelOptions']==1 ]
    X_modelRSU = df[df['modelOptions']==0 ]
    y_modelRSU = df[df['modelOptions']==0 ]

    X_options = df[df['Options']==1 ]
    y_options = df[df['Options']==1 ]

    X_rsu = df[df['Options']==0 ]
    y_rsu  = df[df['Options']==0 ]

    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.scatter(X_rsu['mean_close'], X_rsu['future_mean_close'], color='b', label='RSU')
    ax.scatter(X_options['mean_close'], X_options['future_mean_close'], color='r', label='Options')
    ax.scatter(X_modelOptions['mean_close'], X_modelOptions['future_mean_close'],color='black',s=80, facecolors='none', edgecolors='black', label='Model Options')
    ax.scatter(X_modelRSU['mean_close'], X_modelRSU['future_mean_close'],marker = "x",color='black',s=80, label='Model RSU')

    x_ = np.linspace(X_options['mean_close'].min(), X_options['mean_close'].max(), 100)
    ax.legend(shadow=True)
    ax.set_xlabel('mean_close')
    ax.set_ylabel('future_mean_close')
    ax.set_title('Options or RSU?')
    plt.savefig('images/OptionOrNot4.png')
def find_threshold(X_train, X_test):

    kfold = StratifiedKFold(3,shuffle=False)
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
            y_proba = model.predict_proba(X_train[test_index])[:,1]
            #  Above the model is using predict_proba this returns the 'probability' of a 1
            #  the code below uses this with a differnt threshold to get the actual prediciton
            y_predict = np.array([(lambda z: 1 if z >i else 0)(z) for z in y_proba])
            y_true = y_train[test_index]
            # with some thresholds there are no 1's predicted and an error
            # will be generated
            if(y_predict.sum() != 0):
                F1s.append(f1_score(y_true, y_predict))
                if(np.average(F1s) > bestF1):
                   bestF1 = np.average(F1s)
                   bestthr = i
            F1s = []
    #print('BestF1 and threashold',bestF1,bestthr)
    return(bestthr)

def checkmodel(X_train,y_train):
    X = X_train
    X_const = add_constant(X, prepend=True)
    y = y_train
    logit_model = Logit(y, X_const).fit()
    print(logit_model.summary())
    return(logit_model)


if __name__ == '__main__':
    np.random.seed(42)
    #threshold = thresholdVal
    # This does not need to be processed every time if the
    # Data has not changed
    #df = get_clean_data('data/OracleQuotes.csv',1)
    df = get_clean_data()

    X_train, X_test, y_train, y_test = get_train_test_data(df)
    checkmodel(X_train,y_train)

    threshold = find_threshold(X_train, X_test)

    crossvalidation(X_train, X_test, y_train, y_test, threshold)
    model = runFinalTests(X_train, X_test, y_train, y_test, threshold)
    plot_options(df, model,threshold)
