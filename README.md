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

![image info](images/ss1.png)
![image info](images/ss2.png)
![image info](images/ss3.png)

### Scrubbed Data with future values
If you notice there might not be any data for the exact day 4 years out, so what we really want is the mean price for the week 4 years in advance compared to the mean price/volume/high/low 
![image info](images/msnoSubset.png)

            date  close         volume   open   high      low   datetime  days  Price1y  Price4y
1260  2013/09/23  33.94  26293060.0000  33.80  34.34  33.7500 2013-09-23     0    38.83    48.16
1259  2013/09/24  33.64  18715350.0000  33.83  33.89  33.5900 2013-09-24     1    39.42     0.00
1258  2013/09/25  33.87  19599480.0000  33.76  34.09  33.7000 2013-09-25     2    38.76     0.00
1257  2013/09/26  33.81  15620990.0000  33.98  34.26  33.6900 2013-09-26     3    38.95    47.97
1256  2013/09/27  33.78  14068670.0000  33.63  33.95  33.5101 2013-09-27     4     0.00    47.92

### Scrubbed Data collapsed to weekly

![image info](images/msnoAllRows.png)

![image info](images/confusion.png)
![image info](images/f2.png)

