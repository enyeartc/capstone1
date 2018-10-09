# Models to predict equity retention bonus

Authors: Chris Enyeart 
Web site: https://github.com/enyeartc/capstone1


## Description:
In order to reward and retain talent some companies offer Stock Options, this is the right to purchase stock at a specific price.  If I have an option to buy stock at $40 and the stock is at $50 I make $10.  If the stock is at $30 I have nothing.  You can by options that allow you to make money if the stock drops, but these are not used for retention and will not be discussed here.  As one would guess employees are not incentivized to stay at a company if the options are worthless, so some companies started offering Restricted Stock Units as a way to guarantee a bonus.   This is giving actual stock to an employee at a lower price but less shares than the options.   If you are awarded 1,000 shares you have a choice between 1,000 stock options or 250 RSUs or 500 stock options and 125 RSUs  Vested over 4 years.   

The question really comes down to should an employee choose Restricted Stock Units (RSU's) that have a gaureeted value.  Or choose options where they could make 4 times the ammout.

## Case Study Goal
To accurately create a model that will predict if an employee should choose Options or RSU's.

## Strategy 
Although there are three choices, this project will only see if an employee should choose options or not,  if they do not choose options they will default to RSU's.  This will involve 

## Data   
Initially data looks like the following, but this is only really good if we are predicting data for tomorrow. 
### Initial Data
         date  close         volume   open    high    low
0       9:52   50.85      2,443,215  50.81  51.020  50.71
1  2018/09/21  51.10  50519700.0000  50.83  51.120  50.35
2  2018/09/20  50.43  26973330.0000  49.66  51.075  49.56
3  2018/09/19  49.43  20904240.0000  48.92  49.540  48.66
4  2018/09/18  49.03  33448720.0000  47.51  49.540  47.36

### Scrubbed Data with future values
If you notice there might not be any data for the exact day 4 years out, so what we really want is the mean price for the week 4 years in advance compared to the mean price/volume/high/low 

            date  close         volume   open   high      low   datetime  days  Price1y  Price4y
1260  2013/09/23  33.94  26293060.0000  33.80  34.34  33.7500 2013-09-23     0    38.83    48.16
1259  2013/09/24  33.64  18715350.0000  33.83  33.89  33.5900 2013-09-24     1    39.42     0.00
1258  2013/09/25  33.87  19599480.0000  33.76  34.09  33.7000 2013-09-25     2    38.76     0.00
1257  2013/09/26  33.81  15620990.0000  33.98  34.26  33.6900 2013-09-26     3    38.95    47.97
1256  2013/09/27  33.78  14068670.0000  33.63  33.95  33.5101 2013-09-27     4     0.00    47.92

### Scrubbed Data collapsed to weekly



![image info](images/f1.png)
![image info](images/f2.png)
![image info](images/f3.png)
![image info](images/f4.png)
![image info](images/msno.png)
![image info](images/msnoAllRows.png)
![image info](images/msnoSubset.png)
