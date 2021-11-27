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

boston_df.plot.scatter(x='RM', y='PRICE')
plt.title('Scatter plot\nThe most positive corrolettion the price')
plt.grid()


boston_df.plot.scatter(x='LSTAT', y='PRICE')
plt.title('Scatter plot\nThe most negative corrolettion the price')
plt.grid()


boston_df.plot.scatter(x='AGE', y='PRICE')
plt.title('Scatter plot\nThe most negative corrolettion the price')
plt.grid()

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

"""
in order to define joint distribution needed n*m*2-1 parameters

Explanaation for roy:
    in order to make a joint distribution, we need to fill the table with probabilities for eacvh combnation of values !
    for example, if we have 2 random variables with 2 options for each RV, 
    we will need to fill a 2x2 matrix with probabilities.
    we have a joint distribution probabiltity only if the entire table is full.
    since each table entry is a probability of a single occurunce, and all table entries need to sum up to 1,
    we only need to fill in 3 tables entries, and the last one will be 1 - sum of all others.
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
from sklearn import mixture

GMD_csv_df = pd.read_csv(r"C:\Msc\Git\StatisticsAndDataAnalysis\HW2\GMD_2021.csv", header=None, index_col=0)
samples = GMD_csv_df[1].to_list()

#print(df_gmd.head())
mue_1 = 4
mue_2 = 9
sigma_1 = 0.5
sigma_2 = 0.5
sigma_3 = 1.5

######## way 1 #####
mean_init_array = np.array([[mue_1],[mue_2],[np.mean(samples)]])
weight_2 = 0.25
weight_init = (0.25,weight_2,0.5)
### WAY no' 1 - Using Gaussian mixture with EM ###
print("="*20,"WAY no' 1 using Gaussian mixture with EM","="*20)
# Fit a Gaussian mixture with EM using 2 components
gmm = mixture.GaussianMixture(n_components=3,means_init =mean_init_array ,weights_init=weight_init,  covariance_type='full').fit(GMD_csv_df)

GM_result_df = pd.DataFrame()
GM_result_df['Means'] = gmm.means_.reshape(3,).tolist()
GM_result_df['Weights'] =np.round( gmm.weights_.tolist(),3)
print(GM_result_df)

#ggd
######## way 1 #####


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

sns.distplot(samples,bins=50)
std = np.std(samples)
mean = np.mean(samples)
plt.xlim(xmin= mean-3*std , xmax= mean+3*std)
# The limits are where the cdf is 0.997 by the formula : P(μ-3s ≤ Y ≤ μ+3s) 
plt.grid()
plt.xlabel(f'Sample Values')
plt.title(f'PDF GMD_2021 Data')
plt.show()


"""
#### 3.C
Now assume that the data comes from a Gaussian mixture 
distribution (GMD) with $k=4$.

The given data and parameters stay the same.

Can you estimate the unknown parameters in the two ways
described in section A? Explain.
"""
# 55

mean_init_array = np.array([[mue_1],[mue_2],[np.mean(samples)],[np.mean(samples)]])
weight_2 = 0.25
weight_init = (0.25,weight_2,0.25,0.25)
### WAY no' 1 - Using Gaussian mixture with EM ###
print("="*20,"WAY no' 1 using Gaussian mixture with EM","="*20)
# Fit a Gaussian mixture with EM using 2 components
gmm = mixture.GaussianMixture(n_components=4,means_init =mean_init_array ,weights_init=weight_init,  covariance_type='full').fit(GMD_csv_df)

GM_result_df = pd.DataFrame()
GM_result_df['Means'] = gmm.means_.reshape(4,).tolist()
GM_result_df['Weights'] =np.round( gmm.weights_.tolist(),3)
print(GM_result_df)
"""
#### 3.D
Describe two ways for generating data for a GMM random variable with:
* centers at  $\mu_1=3, \mu_2=7, \mu_3=10$
* $\sigma_1=\sigma_2=\sigma_3=1$
* $w_1=w_2=w_3=0.33$
"""
#### way 1

"""
Like we did in the first question-
1. create a normal distrubution to each geonosian with size of N
2. create an empty GMD array
3. do N times:
    - randomize a P(probability) that considerate the weight of each distrubution
    - add to the GMD array a value from one of the normal distrubution
    
#### way 2
1. Genarate 2 normal distrubution with size of 500
2. join  the 2 vectors into one which represent the GMD
"""
import scipy.stats
import numpy as np
import numpy.random
import scipy.stats as ss
import matplotlib.pyplot as plt

mue1 = 3
mue2 = 7
mue3 = 10

sigma1 = sigma2 = sigma3 = 1

w1 = w2 = w3 = 0.33


n = 10000
numpy.random.seed(0x5eed)
# Parameters of the mixture components
norm_params = np.array([[mue1, sigma1],
                        [mue2, sigma2],
                        [mue3, sigma3]])
n_components = norm_params.shape[0]
# Weight of each component, in this case all of them are 1/3
weights = np.ones(n_components, dtype=np.float64) / 3.0
# A stream of indices from which to choose the component
mixture_idx = numpy.random.choice(len(weights), size=n, replace=True, p=weights)
# y is the mixture sample
y = numpy.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx),
                   dtype=np.float64)

# Theoretical PDF plotting -- generate the x and y plotting positions
xs = np.linspace(y.min(), y.max(), 200)
ys = np.zeros_like(xs)

for (l, s), w in zip(norm_params, weights):
    ys += ss.norm.pdf(xs, loc=l, scale=s) * w

sns.distplot(y,bins=100)
plt.title('way A  - to create GMD')
"""
plt.plot(xs, ys)
plt.hist(y, bins=100)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
"""


plt.figure()
norm1 = np.random.normal(mue1, sigma1,1000)
norm2 = np.random.normal(mue2, sigma2,1000)
norm3 = np.random.normal(mue3, sigma3,1000)

new_norm2 = []
new_norm2.extend(norm1)
new_norm2.extend(norm2)
new_norm2.extend(norm3)
sns.distplot(new_norm2, bins=100)
plt.title('way B  - to create GMD')

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

mean = 70000
std = 30000
desire_sallary = 50000
less_than_50k = scipy.stats.norm(mean,std).cdf(desire_sallary)




# ffd


"""
#### 4.B
What percent of people earn between 45,000 RCU and 65,000 RCU?

"""
up_desire_sallary = 65000
down_desire_sallary = 45000

up_cdf_result = scipy.stats.norm(mean,std).cdf(up_desire_sallary)
down_cdf_result = scipy.stats.norm(mean,std).cdf(down_desire_sallary)

percent_to_sallary_between_2_values = up_cdf_result - down_cdf_result



# fff

"""
#### 4.C
What percent of people earn more than 70,000 RCU?
"""
more_than_desire_sallary = 70000

percent_more_than_desire_sallary = 1-scipy.stats.norm(mean,std).cdf(more_than_desire_sallary)




### 

"""
#### 4.D
The company has 1000 employees. How many employees in the 
company do you expect to earn more than 140,000 RCU?
"""
more_than_desire_sallary = 140000
amount_of_employees = 1000
percent_more_than_desire_sallary = 1-scipy.stats.norm(mean,std).cdf(more_than_desire_sallary)
desire_amount_of_employees = np.ceil(amount_of_employees*percent_more_than_desire_sallary)

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



