import pandas as pd
import numpy as np
import csv
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn import svm, datasets
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

def get_model_profits(model, cost_benefit, X_train, X_test, y_train, y_test):
    """Fits passed model on training data and calculates profit from cost-benefit
    matrix at each probability threshold.

    Parameters
    ----------
    model           : sklearn model - need to implement fit and predict
    cost_benefit    : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
    X_train         : ndarray - 2D
    X_test          : ndarray - 2D
    y_train         : ndarray - 1D
    y_test          : ndarray - 1D

    Returns
    -------
    model_profits : model, profits, thresholds
    """
    model.fit(X_train, y_train)
    predicted_probs = model.predict_proba(X_test)[:, 1]
    predicted = model.predict(X_test)
    #print(predicted_probs)
    # print(predicted)
    # print(y_train)
    #print(y_test)

    profits, thresholds = profit_curve(cost_benefit, predicted_probs, y_test)

    return profits, thresholds
def plot_model_profits(model_profits, save_path='images/profit.png'):
    """Plotting function to compare profit curves of different models.

    Parameters
    ----------
    model_profits : list((model, profits, thresholds))
    save_path     : str, file path to save the plot to. If provided plot will be
                         saved and not shown.
    """
    for model, profits, _ in model_profits:
        percentages = np.linspace(0, 100, profits.shape[0])
        plt.plot(percentages, profits, label=model.__class__.__name__)

    plt.title("Profit Curves")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.legend(loc='best')
    plt.savefig(save_path)

def make_clean_Options_df(df):
    #df = df[['Options','ma2_mean']]
    df = df[['Options','mean_close','ma_2']]

    #df = df[['Options','mean_volume']]
    return (df)

# def get_clean_data():
#     df = pd.read_csv('data/clean_small2.csv')
#     df = df[['Options','ma1_mean']]
#     return(df)

def standard_confusion_matrix(y_true, y_pred):
    """Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------
    Parameters
    ----------
    y_true : ndarray - 1D
    y_pred : ndarray - 1D

    Returns
    -------
    ndarray - 2D
    """

    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])
def profit_curve(cost_benefit, predicted_probs, labels):
    """Function to calculate list of profits based on supplied cost-benefit
    matrix and prediced probabilities of data points and thier true labels.

    Parameters
    ----------
    cost_benefit    : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
    predicted_probs : ndarray - 1D, predicted probability for each datapoint
                                    in labels, in range [0, 1]
    labels          : ndarray - 1D, true label of datapoints, 0 or 1

    Returns
    -------
    profits    : ndarray - 1D
    thresholds : ndarray - 1D
    """
    n_obs = float(len(labels))
    maybe_one = [] if 1 in predicted_probs else [1]
    thresholds = maybe_one + sorted(predicted_probs, reverse=True)
    profits = []
    for threshold in thresholds:
        y_predict = predicted_probs >= threshold
        #print('labels',labels)
        confusion_matrix = standard_confusion_matrix(labels, y_predict)
        #print(confusion_matrix)
        #print(threshold)
        threshold_profit = np.sum(confusion_matrix * cost_benefit) / n_obs
        #print('threshold_profit',threshold_profit)
        profits.append(threshold_profit)
    return np.array(profits), np.array(thresholds)
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

def processcsv(filepath, cost_benefit):
    """Main function to test profit curve code.

    Parameters
    ----------
    filepath     : str - path to find Options.csv
    cost_benefit  : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
    """
    X_train, X_test, y_train, y_test = get_train_test(filepath)
    #models = [RF(n_estimators = 10), LR(solver='liblinear'), GBC(), SVC(probability=True)]
    models = [LR(solver='liblinear')]
    model_profits = []
    for model in models:
        profits, thresholds = get_model_profits(model, cost_benefit,
                                                X_train, X_test,
                                                y_train, y_test)
        model_profits.append((model, profits, thresholds))
    plot_model_profits(model_profits)

    max_model, max_thresh, max_profit = find_best_threshold(model_profits)
    max_labeled_positives = max_model.predict_proba(X_test) >= max_thresh
    plotROC(y_test, max_model.predict_proba(X_test)[:,1])
    # Add a line for just a 1

    proportion_positives = max_labeled_positives.mean()
    reporting_string = ('Best model:\t\t{}\n'
                        'Best threshold:\t\t{:.2f}\n'
                        'Resulting profit:\t{}\n'
                        'Proportion positives:\t{:.2f}')
    print(reporting_string.format(max_model.__class__.__name__, max_thresh,
                                  max_profit, proportion_positives))

def find_best_threshold(model_profits):
    """Find model-threshold combo that yields highest profit.

    Parameters
    ----------
    model_profits : list((model, profits, thresholds))

    Returns
    -------
    max_model     : str
    max_threshold : float
    max_profit    : float
    """
    max_model = None
    max_threshold = None
    max_profit = None
    for model, profits, thresholds in model_profits:
        max_index = np.argmax(profits)
        if not max_model or profits[max_index] > max_profit:
            max_model = model
            max_threshold = thresholds[max_index]
            max_profit = profits[max_index]
    return max_model, max_threshold, max_profit
def plotROC(y_true, y_prob_H, filelocation = 'images/plotROC.png',title = 'ROC curve'):
    print("in plotRoc",filelocation,title)
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

    # ax2 = plt.gca().twinx()
    # ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')
    # ax2.set_ylabel('Threshold',color='r')
    # ax2.set_ylim([thresholds[-1],thresholds[0]])
    # ax2.set_xlim([fpr[0],fpr[-1]])
    plt.savefig(filelocation)
def oversample0(X_train, y_train):
    # Quickly Oversample data with more RSU

    for Xt,yt in zip(X_train, y_train):
        if(yt == 0):
            X_train = np.append(X_train,[Xt], axis=0)
            print('After',X_train)
            y_train = np.append(y_train,yt)
    return(X_train, y_train)

def crossvalidation(filepath):
    raw_df = pd.read_csv(filepath)
    Options_df = make_clean_Options_df(raw_df)
    print (Options_df.head())
    y = Options_df.pop('Options').values
    X = Options_df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,test_size=0.33)
    # After the split
    #X_train,y_train = oversample0(X_train,y_train)


    kfold = StratifiedKFold(5,shuffle=False)

    accuracies = []
    precisions = []
    recalls = []
    sumofy = []
    sumofy2 = []
    F1s = []

    for train_index, test_index in kfold.split(X_train,y_train):
        model = LR(solver='liblinear')
        model.fit(X_train[train_index], y_train[train_index])
        #y_predict = model.predict(X_train[test_index])
        y_proba = model.predict_proba(X_train[test_index])[:,1]
        #  Above the model is using predict_proba this returns the 'probability' of a 1
        #  the code below uses this with a differnt threshold to get the actual prediciton
        y_predict = np.array([(lambda z: 1 if z >0.45177236 else 0)(z) for z in y_proba])
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


        # print ("accuracy:", np.average(accuracies))
        # print ("precision:", np.average(precisions))
        # print ("recall:", np.average(recalls))
        # print ("sumofyPridict:", np.average(sumofy))
        # print ("sumofyTrue:", np.average(sumofy2))
        # print ('F1', np.average(F1s))

    y_proba = model.predict_proba(X_train)[:,1]
    y_predict = np.array([(lambda z: 1 if z >0.6443742 else 0)(z) for z in y_proba])
    #print(y_train, y_predict)
    plotROC(y_train, y_proba,"images/plotROC_Training.png","ROC Curve Training")

    y_proba = model.predict_proba(X_test)[:,1]
    y_predict = np.array([(lambda z: 1 if z >0.6443742 else 0)(z) for z in y_proba])
    print("__________________test")

    #print(y_proba,y_test, y_predict)
    print("Threashold 0.6443742")
    plotROC(y_test, y_proba,'images/ROC_Test.png',"ROC Curve Test")
    print ('Test accuracy_score',accuracy_score(y_test, y_predict))
    print ('Test precision_score',precision_score(y_test, y_predict))
    print ('Test recall_score',recall_score(y_test, y_predict))
    print ('Test f1_score',f1_score(y_test, y_predict))
    print('---------------------summmary')
    print(raw_df.Options.value_counts())
    print('train sum',sum(y_train),'train len',len(y_train))
    zipped = set(zip(y_proba,y_test, y_predict))
    cw = csv.writer(open("zipped.csv",'w'))
    cw.writerow(zipped)
    y_proba = model.predict_proba(X_train)[:,1]
    y_predict = np.array([(lambda z: 1 if z >0.0 else 0)(z) for z in y_proba])
    print('------All ones')
    print(len(y_proba),len(y_train),len(y_predict))
    #print(y_proba,y_train, y_predict)
    # print("Threashold 0.6443742")
    plotROC(y_train, y_proba,'images/plotAll1.png','ROC Curve All 1')
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
def crossvalidationThr(filepath):
    raw_df = pd.read_csv(filepath)
    Options_df = make_clean_Options_df(raw_df)
    print (Options_df.head())
    y = Options_df.pop('Options').values
    X = Options_df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y,test_size=0.33)
    # After the split
    #X_train,y_train = oversample0(X_train,y_train)


    kfold = StratifiedKFold(5,shuffle=False)

    accuracies = []
    precisions = []
    recalls = []
    sumofy = []
    sumofy2 = []
    F1s = []
    thr = [0.27411032 ,0.45177236 ,0.43848543 ,0.57502369 ,0.52727048, 0.26294993,
 0.3734106,  0.55862627 ,0.35298962, 0.55236403, 0.86459271, 0.47550826,
 0.36730173, 0.69804784 ,0.42914183, 0.56947256, 0.59538206, 0.58619825,
 0.73952725, 0.76668073 ,0.7893238 , 0.65972535, 0.77526139, 0.43201526,
 0.73858921, 0.60181424 ,0.43968636, 0.75375776, 0.78417738, 0.53074894,
 0.75953991, 0.62555268 ,0.48809615, 0.67150554, 0.52648925]
    bestthr = 0
    bestF1  = 0
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
            print(y_proba)
            print('Predict',y_predict)
            print('True   ',y_true)
            accuracies.append(accuracy_score(y_true, y_predict))
            precisions.append(precision_score(y_true, y_predict))
            recalls.append(recall_score(y_true, y_predict))
            sumofy.append(len(y_predict)-y_predict.sum())
            sumofy2.append(len(y_true)-y_true.sum())
            F1s.append(f1_score(y_true, y_predict))


            print ("accuracy:", np.average(accuracies))
            print ("precision:", np.average(precisions))
            print ("recall:", np.average(recalls))
            print ("sumofyPridict:", np.average(sumofy))
            print ("sumofyTrue:", np.average(sumofy2))
            print ('F1', np.average(F1s))
            if(np.average(F1s) > bestF1):
               bestF1 = np.average(F1s)
               bestthr = i
            accuracies = []
            precisions = []
            recalls = []
            sumofy = []
            sumofy2 = []
            F1s = []
    print('BestF1 and threashold',bestF1,bestthr)
if __name__ == '__main__':
    #test_cost_benefit()
    #df = get_clean_data()
    np.random.seed(42)
    plt.figure(figsize=(16, 8))

    costben = np.array([[1,-5],[-5,1]])
    #processcsv('data/clean_small2.csv', costben)
    plot_options('data/clean_small2.csv')
    #plt.savefig('images/profitCurve1.png')
    crossvalidation('data/clean_small2.csv')
    #crossvalidationThr('data/clean_small2.csv')
