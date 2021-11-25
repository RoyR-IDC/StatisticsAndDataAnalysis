"""
### IDs:
Insert yours IDs to the cell below

ID #1:

ID #2:

"""




# ------------------------------------------------------------------------------------------------------------


"""

## Read the following instructions carefully:

1. This jupyter notebook contains all the step by step instructions needed 
for this exercise.
1. You are free to add cells.
1. Write your functions and your answers in this jupyter notebook only.
1. Answers to theoretical questions should be written in **markdown cells 
(with $\LaTeX$ support)**.
1. Submit this jupyter notebook only using your ID as a filename. 
Not to use ZIP or RAR. For example, your Moodle submission
 file name should look like this (two id numbers): `123456789_987654321.ipynb`.

"""
# ------------------------------------------------------------------------------------------------------------


"""
### Question 1 - Data exploration and visialization - practical
Load Boston dataset from sklearn.
Explore the data. follow th instructions below and make sure to support
 your answers with proper outputs and plots.
When plotting, pay close attention to the range of the axis, and include
 axis labels and a title for the figure.

1. describe the dataset. How many samples does it contain? 
How many features? What isis the data type for each variable?
2. Produce a histogram and a boxplot of the nitric oxides concentration. 
describe the distribution.
3. Produce a correlation matrix of all the features. 
[Are there any correlated features?
 Can you identify one feature with unusual behaviour?
4. Select the 2 pairs of features with the highest correlation
 (positive or negative) and plot 2 scatter plots with marginal histograms (JointPlot). 
5. Produce a cumulative histogram of the age variable and 
add two horizontal lines on the first and third quartile (on the cumulative count)
6. Identify and report 2 “interesting” trends in the data. 
No need to provide statistical confidence at this point. 
"""
#SETUP
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn import mixture
plt.style.use('default') # You need to "reset" befoe you switch styles.
plt.style.use('seaborn-pastel')
mpl.rcParams['figure.figsize'] = [6, 4]
from sklearn.datasets import load_boston
boston = load_boston()

X = boston.data
columns = list(boston.feature_names)
y = boston.target
boston_df = pd.DataFrame(np.concatenate((X, y[:, np.newaxis]), axis=1), \
                         columns= columns + ['PRICE'])
#boston_df.head()
print(boston_df.to_markdown())

#1. describe the dataset. How many samples does it contain? 
#   How many features? What isis the data type for each variable?

#
amount_of_feature = boston_df.shape[1]-1
amount_of_sampels = boston_df.shape[0]
print('The amount of sampels is: ' + str(amount_of_sampels))
print('The amount of feature is: ' + str(amount_of_feature))
print('data type for each feature:')
boston_df.info(verbose=True)

#2. Produce a histogram and a boxplot of the nitric oxides concentration. 
#   describe the distribution.
boston_df_col = boston_df.columns.to_numpy()

plt.figure()
ax = sns.histplot(x="NOX", data=boston_df)
plt.title('histogram plot of NOX feature')
plt.grid()
NOX_feature = boston_df['NOX'].to_list()
plt.axvline(x = 0.44, color= 'r')
plt.axvline(x = 0.66, color= 'r')

np.mean(NOX_feature)
np.std(NOX_feature)
np.median(NOX_feature)

"""
we can see from the histogram plot that most of the data in the range 
of [mean-std,mean+std]
in our case the mean is 0.55 and the std is 0.11 -> [0.44,0.66]
"""

plt.figure()
ax = sns.boxplot(x="NOX", data=boston_df)
plt.title('box plot of NOX feature')
#plt.grid()
"""
the box plot show the precentiles:
    1. the first line describe the 0% percentile (the min value)
    2. the secound line describe the 25% percentile (Q1)
    3. the third line describe the 50% percentile (median) 
    4. the forth line describe the 75% percentile (Q3) 
    5. the fivth line describe the 100% percentile (the max value) 
from our box plot we can infer that the IQR(inter quartile range) is
between ~0.44 to ~0.66 which correlate with the histogram conclution 
"""

#3. Produce a correlation matrix of all the features. 
#   [Are there any correlated features?

corr_array = boston_df.corr().to_numpy()
np.fill_diagonal(corr_array, 0)
max_corr_row_col = np.where(corr_array == np.max(corr_array))[0]
min_corr_row_col = np.where(corr_array == np.min(corr_array))[0]
min_cor_features = boston_df_col[min_corr_row_col].tolist()
max_cor_features = boston_df_col[max_corr_row_col].tolist()





plt.figure()
sns.heatmap(boston_df.corr(),cmap='coolwarm', vmin=-1, vmax=1)
plt.title('correlation matrix of all the features.')

string = 'the maximum correlation is between each feature with himself in the digonal'
print(string)
string = 'The min corroletion found to be betwenn feature ' + min_cor_features[0] + ' to feature ' \
+ min_cor_features[1]
print(string)

string = 'The max corroletion found to be betwenn feature ' + max_cor_features[0] + ' to feature ' \
+ max_cor_features[1]
print(string)

string = 'The unusual feature is the CHAS, it can be seen from the heat map,' + \
        'that he has no correlation with other features'
print(string)
#4. Select the 2 pairs of features with the highest correlation
# (positive or negative) and plot 2 scatter plots with marginal histograms (JointPlot). 



ax= sns.jointplot(x=boston_df[max_cor_features[0]], y=boston_df[max_cor_features[1]],\
              marginal_kws=dict(bins=30))
ax.fig.suptitle('Joint plot between ' + max_cor_features[0]+ ', '+ max_cor_features[1])

ax= sns.jointplot(x=boston_df[min_cor_features[0]], y=boston_df[min_cor_features[1]],\
              marginal_kws=dict(bins=30))
ax.fig.suptitle('Joint plot between ' + min_cor_features[0]+ ', '+ min_cor_features[1])


#5. Produce a cumulative histogram of the age variable and 
#   add two horizontal lines on the first and third quartile (on the cumulative count)

plt.figure()
ax = sns.histplot(x="AGE", data=boston_df, cumulative=True, stat='density')
plt.title('cumulative histogram plot of AGE feature')
cumsum = boston_df['AGE'].to_numpy().cumsum()
plt.axhline(y= 0.25, color = 'r')
plt.axhline(y= 0.75, color = 'r')
plt.grid()

#6. Identify and report 2 “interesting” trends in the data. 
#   No need to provide statistical confidence at this point. 

sns.scatter(x='index', y= 'RM', hue ='PRICE', data= boston_df.reset_index(0))
boston_df.plot.scatter(x='RM', y='PRICE')
"""


"""
"""
the box plot show the precentiles:
    1. the first line describe the 0% percentile (the min value)
    2. the secound line describe the 25% percentile (Q1)
    3. the third line describe the 50% percentile (median) 
    4. the forth line describe the 75% percentile (Q3) 
    5. the fivth line describe the 100% percentile (the max value) 
from our box plot we can infer that the IQR(inter quartile range) is
between ~0.44 to ~0.66 which correlate with the histogram conclution 
"""
"""
# ------------------------------------------------------------------------------------------------------------

### Question 2 - Independence and conditional independence

#### 2.A
Let $\ X, Y \ $ and $Z$  be discrete random variables with $\ n, m \ $ 
and $k=2$ possible outcomes respectivley.

How many parameters define the joint distribution of $\ X, Y \ $ and $Z$?
"""



#ddd


"""
#### 2.B
For the same random variables from the previous section, 
how many parameters define the joint distribution of $\ X, Y \ $ 
and $Z$ if we now know that they are independent?
"""



# ddd

"""
#### 2.C
For the same random variables from the previous section, 
how many parameters define the joint distribution of $\ X, Y \ $ 
and $Z$ if we now know that $X$ and $Y$ are conditionaly independent given $Z$?
"""


#dfff



"""
#### 2.D
Give an example for a joint distribution of $\ X, Y \ $ 
and $Z$ where $X$ and $Y$ are NOT conditionally independent given $Z$, 
but $X$ and $Y$ are (unconditionally) independent.
Where $X$ and $Y$ are standard normal distribution ($N(0, 1)$).

"""

# ------------------------------------------------------------------------------------------------------------

# ddd
"""
### Question 3 - Gaussian mixtures – parameter estimation and generation 

Consider the data provided in GMD_2021.csv
Assume that the data comes from a Gaussian mixture distribution (GMD) 
with $k=3$. Furthermore, assume that
 $\mu_{1}=4, \mu_{2}=9, \sigma_{1}=\sigma_{2}=0.5, \sigma_3=1.5$ and $w_2=0.25$.

Read the data and answer the following questions.

"""


#ggd


"""
#### 3.A
Provide an estimate for the other parameters of the distribution in two
 different ways.
"""

## fff

"""
#### 3.B
Plot a graph of the pdf of the distribution you inferred.
 Select adequate limits for the axes for this plot and explain your decision.
"""
#fff


"""
#### 3.C
Now assume that the data comes from a Gaussian mixture 
distribution (GMD) with $k=4$.

The given data and parameters stay the same.

Can you estimate the unknown parameters in the two ways
described in section A? Explain.
"""
# 55


"""
#### 3.D
Describe two ways for generating data for a GMM random variable with:
* centers at  $\mu_1=3, \mu_2=7, \mu_3=10$
* $\sigma_1=\sigma_2=\sigma_3=1$
* $w_1=w_2=w_3=0.33$
"""


# gggg
"""
#### 3.E
Use one of the above approaches to generate 1000 points and plot
 a histogram of the result (decide on bins, axes etc)

"""

## ffff


"""
#### 3.F
Use the other one to generate 1000 more points and draw two comparative histograms.
"""
## ffff
# ------------------------------------------------------------------------------------------------------------

"""
### Question 4 - Normally distributed salaries
The annual salaries of employees in a large Randomistan company
 are approximateley normally distributed with a mean of 70,000 RCU 
 and a standard deviation of 30,000 RCU.

#### 4.A
What percent of people earn less than 50,000 RCU?
"""



# ffd


"""
#### 4.B
What percent of people earn between 45,000 RCU and 65,000 RCU?

"""



# fff

"""
#### 4.C
What percent of people earn more than 70,000 RCU?
"""



### 

"""
#### 4.D
The company has 1000 employees. How many employees in the 
company do you expect to earn more than 140,000 RCU?
"""

# ------------------------------------------------------------------------------------------------------------

"""
### Question 5 - Coupon collector
Let $T_{N}$ denote the waiting time for full single coupon
 collection with N different equiprobable coupon types
"""

"""
#### 5.A
Write code to compute the exact value of $E(T_{N})$
"""



"""
#### 5.B
Write code to compute the exact value of $V(T_{N})$
"""



"""
#### 5.C
Write code to exactly compute $P(T_{30}>60)$
"""



"""
#### 5.D
Use Chebicheff to provide a bound for the probability from C
and compare the results
"""
# ------------------------------------------------------------------------------------------------------------



