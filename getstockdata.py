import numpy as np
import pandas as pd

import missingno as msno
import matplotlib.pyplot as plt

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
    ''' This funciton will create a file clean_small2 which will be
        a new csv that will contain

    '''
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
