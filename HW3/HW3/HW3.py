"""
Provide example datapoints matching the following descriptions. Each example should be constructed over n=50 datapoints. Provide a table description of the example data as well as a jointplot (see example below).
If you think that the situation described is impossible then clearly explain why (you don’t need to give a rigorous proof).<br>
Pearson(x,y) = Pearson correlation<br>
τ(x,y) = Kendall rank correlation<br>
ρ(x,y) = Spearman rank correlation <br>
<img src="jointplot.png">
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
import time
from scipy import stats as st # for pearsonr, kendalltau, spearmanr etc.
from scipy.stats import multivariate_normal as mn
from numpy.linalg import matrix_power
from scipy.stats import rv_discrete
import warnings

def print_table_and_jointplot(x, y, d = True):
    # table description of the example data
    table = pd.DataFrame(np.vstack((x,y)).T, columns = ['x', 'y'])

    if d != False:
        table['d_i'] = table['x'] - table['y']

    print(table)
    p = sns.jointplot(data=table, x=x, y=y, marker='*')
    p.fig.suptitle('x & y values JointPlot', fontsize=25)
    p.fig.subplots_adjust(top=0.9) # Reduce plot to make room for title
    p.set_axis_labels(xlabel = 'X', ylabel= 'Y')


n = 50 # 50 datapoints

x = np.zeros(n)
y = np.zeros(n)

#
x[n-2] = 1e-5
y[n-2] = 1e-5

# these are the last values we will exclude
x[n-1] = -100
y[n-1] = 100

all_data_correlation = st.pearsonr(x, y)[0]
partial_correlation = st.pearsonr(x[:n-1], y[:n-1])[0]




S1 = 'Pearson correlation between x & y all sampels ' + str(np.round(all_data_correlation,3))
S2 = 'Pearson correlation between x & y all sampels except last ellement ' + str(np.round(partial_correlation,3))
print(S1 + '\n' + S2)

print_table_and_jointplot(x, y, d=False)
"""
Data with τ(x,y) > ρ(x,y) + 0.45
"""
n = 50 # 50 datapoints


x = np.arange(n) + 1
y = np.empty(n)

y[0:n//2] = (np.arange(n//2) + 1) + 50
y[n//2:] = (np.arange(n//2) + 1)

kendall_correlation = st.kendalltau(x, y)[0]
spearman_correlation = st.spearmanr(x, y)[0]
diff_of_correlations = kendall_correlation-spearman_correlation



S1 = 'kendall correlation between x & y all sampels ' + str(np.round(kendall_correlation,3))
S2 = 'spearman correlation between x & y all sampels ' + str(np.round(spearman_correlation,3))
S3 = 'kendall correlation - spearman correlation is ' + str(np.round(diff_of_correlations,3)) +', which is greater than 0.45'

print(S1 + '\n' + S2 +'\n' +S3)

print_table_and_jointplot(x, y)

"""
we design x,y such that they will assembel 2 linear line with a same positive slope
by doing that we assume to achieve negative correlaction value, because of the way we build
the two line, the kendal score will be small negative number (there is more discordant than cordant)
/tho = C-D/(n/2)

"""



"""
#### 1.C
Data with τ(x,y) < ρ(x,y) – 0.45
"""


x = np.arange(n) + 1
y = np.empty(n)

y[0:n//2] = (np.arange(n//2) + 1)[::-1]
y[n//2:] = (np.arange(n//2) + 1)[::-1]+ 100

kendall_correlation = st.kendalltau(x, y)[0]
spearman_correlation = st.spearmanr(x, y)[0]
diff_of_correlations = kendall_correlation-spearman_correlation



S1 = 'kendall correlation between x & y all sampels ' + str(np.round(kendall_correlation,3))
S2 = 'spearman correlation between x & y all sampels ' + str(np.round(spearman_correlation,3))
S3 = 'kendall correlation - spearman correlation is ' + str(np.round(diff_of_correlations,3)) +', which is greater than -0.45'


print(S1 + '\n' + S2 +'\n' +S3)

print_table_and_jointplot(x, y)


"""
we design x,y such that they will assembel 2 linear line with a same negative slope
by doing that we assume to achieve positive correlaction value, because of the way we build
the two line, the kendal score will be small positive number (there is more cordant than discordant)
/tho = C-D/(n/2)

"""

"""
#### 1.D
Data with Pearson(x,y) < ρ(x,y) – 0.6
"""
x = np.linspace(start=1, stop=50, num=50)
y = 10**x # exponential function

pearson_correlation = st.pearsonr(x, y)[0]
spearman_correlation = st.spearmanr(x, y)[0]
diff_of_correlations = spearman_correlation - pearson_correlation

S1 = 'pearson_correlation between x & y all sampels ' + str(np.round(pearson_correlation,3))
S2 = 'spearman correlation between x & y all sampels ' + str(np.round(spearman_correlation,3))
S3 = 'spearman correlation - pearson_correlation -  is ' + str(np.round(diff_of_correlations,3)) +', which is greater than 0.6'
print(S1 + '\n' + S2 +'\n' +S3)
print_table_and_jointplot(x, y)

"""
#### 1.E
Data with Pearson(x,y) > ρ(x,y) + 1.2
"""

x = np.arange(n) + 1
y = np.linspace(50, 1, num=n)

x[n-1] = 10000
y[n-1] = 10000

pearson_correlation = st.pearsonr(x, y)[0]
spearman_correlation = st.spearmanr(x, y)[0]
diff_of_correlations = pearson_correlation- spearman_correlation

S1 = 'pearson_correlation between x & y all sampels ' + str(np.round(pearson_correlation,3))
S2 = 'spearman correlation between x & y all sampels ' + str(np.round(spearman_correlation,3))
S3 = 'pearson_correlation - spearman correlation is ' + str(np.round(diff_of_correlations,3)) +', which is greater than 1.2'
print(S1 + '\n' + S2 +'\n' +S3)
print_table_and_jointplot(x, y)
"""
we design array that contain negative linear line, in order to convert spearman to
be negative, we take that last ellement to be large dramiticaly in both axis, by that we
convert person correlation to be positive, and sprearman correlation to negative, because
all the difference until n-1 ellement are small, and the last difference is dramitcly positive
"""


"""
#### 1.F
Data with τ(x,y) < ρ(x,y) – 1.2
"""

"""
This scenario is not possible
our intuition:
in order that the differance between spearman to kendal to be greater than 1.2,
the kendal and spreaman needed to be in differnt signs, Kendall and Spearman will have can have different signs,
however the difference between there score base on privious 1.b,1.c base on our exipirements, the difference
between tham cannot pass 0.5, therefore it can't pass 1.2
if we will look on the extreame of spearman an kendal correlation, they needed to behave the same:
if Spearman = 1.0 --> all the data is monotonically increasing and all the ranks agree, --> in Kendall notation
all the pairs are concordant, so Kendall will also equal 1
if Kendall = -1.0 --> all the pairs are discordant--> all the data is monotonically decreasing -->
Spearman will also equal -1
"""

"""
Perform data analysis on the UCI Heart Disease Dataset
References:
1. Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., Guppy, K., Lee, S., & Froelicher, V. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. American Journal of Cardiology, 64,304--310.
2. David W. Aha & Dennis Kibler. "Instance-based prediction of heart-disease presence with the Cleveland database."
3. Gennari, J.H., Langley, P, & Fisher, D. (1989). Models of incremental concept formation. Artificial Intelligence, 40, 11--61.

Use the following links to find the details of the data:
1. https://archive.ics.uci.edu/ml/datasets/heart+disease
2. https://www.kaggle.com/ronitf/heart-disease-uci

In the follwong cells write a report for your analysis. In your report provide a clear description of the analysis methods and of the results. This should include a graphical representation of the results and the conclusions. Provide p-values or other indicators of the statistical significance where appropriate. <br>
Design your report to be concise but still cover interesting findings.

There are missing values in the data. Find them and impute them by using the median (for numerical features) or the mode (for categorical features) value of the relevant feature (column).
Address the following questions:
* Using confidence intervals determine for which numerical features you can state with confidence of 95% that the healthy population (target = 0) mean is larger/smaller than the disease population (target = 1) mean.
* Draw histograms for each numerical feature. Inspect the observed distributions and then use MLE to plot, on the same figures, fits of the distributions.
* For each pair of numerical feature, calculate correlations and indicate whether you find them significant. For select pairs, with significant correlations, draw joint plot with marginal histograms (see Seaborn joint plot) and find bivariate normal fits. Then use the example code below to draw plots of the bivariate pdfs (you may edit the code as you see fit).
* Are there pairs that are significantly correlated in males but not in femalees? The opposite? How about healthy vs disease? Can you graphically represent this?
* For each numerical feature, except age, plot the distribution for this feature against bins of age. In each bin provide a split violin plot, with different colors for healthy and disease.

Suggest, state and address at least one original question.

"""
# Example code for bivariate pdfs
from scipy.stats import multivariate_normal as mn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random
import time
from scipy import stats as st # for pearsonr, kendalltau, spearmanr etc.
from scipy.stats import multivariate_normal as mn
from numpy.linalg import matrix_power
from scipy.stats import rv_discrete
import warnings
def plot_2d_Gaussian_pdf(means, cov, feature_name1, feature_name2):
    n = 100
    x1 = np.linspace(means[0] - 3 * np.sqrt(cov[0][0]), means[0] + 3 * np.sqrt(cov[0][0]), n)
    x2 = np.linspace(means[1] - 3 * np.sqrt(cov[1][1]), means[1] + 3 * np.sqrt(cov[1][1]), n)
    x1_v, x2_v = np.meshgrid(x1, x2)
    Xgrid = np.vstack([x1_v.ravel(), x2_v.ravel()]).T
    Y = mn.pdf(Xgrid, means, cov)
    fig, ax = plt.subplots()
    ax.pcolorfast(x1, x2, Y.reshape(x1_v.shape), alpha=0.5, cmap='Blues')
    ax.contour(x1_v, x2_v, Y.reshape(x1_v.shape),
                alpha=0.3, colors='b')
    ax.axis('equal')
    ax.grid(alpha=0.2)
    plt.title('2d Gaussian pdf between ' + feature_name1 + ' to ' + feature_name2)
    plt.xlabel(feature_name1)
    plt.ylabel(feature_name2)
    plt.show()

means = [3, 2]
cov = [[1, 0.5], [0.5, 0.8]]
plot_2d_Gaussian_pdf(means, cov , 'x', 'y')
df = pd.read_csv(r"D:\HW3\heart.csv", index_col=False, sep='\t')

df_describe = df.describe(include ='all')

df_median = df_describe.loc['50%']
#df_median = df_describe.loc['mean']

df_mean = df_describe.loc['mean']


df.head()
df_info   = df.info(verbose = True)

categorical_columns  = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
df_columns_list = df.columns.to_list()
numerical_feature_columns = list(set(df_columns_list)-set(categorical_columns))

for i_col in numerical_feature_columns:
    df[i_col].fillna(df_median[i_col], inplace = True)
for i_col in categorical_columns:
    mode = scipy.stats.mode(df[i_col].to_list()).mode[0]
    df[i_col].fillna(mode, inplace = True)

numerical_feature_df = df[numerical_feature_columns]
categorical_feature_df = df[categorical_columns]

sns.pairplot(numerical_feature_df)

for i_col in numerical_feature_columns:
    plt.figure()
    #feature_sample_array = numerical_feature_df[i_col].to_numpy()
    #aprox_mue = np.mean(feature_sample_array)
    #one_d_n = (1/numerical_feature_df.shape[0])
    #square_summation = np.sum(np.square((feature_sample_array-aprox_mue)))
    #aprox_sigma = np.sqrt(one_d_n*square_summation)
    sns.distplot(numerical_feature_df[i_col], fit = scipy.stats.norm)
    plt.legend()
    plt.grid()
    plt.title(i_col + ' feature histogram\nMLE plot')

healthy = (categorical_feature_df['target']==1)
disease = ~healthy

healthy_numerical_feature_df = numerical_feature_df.loc[healthy] # seek
disease_numerical_feature_df = numerical_feature_df.loc[disease]
alpha =0.05
summary_string = 'confidence interval for all feature, aprove or decline null model:\n'
for i_col_index in range(numerical_feature_columns.__len__()):
    i_col = numerical_feature_columns[i_col_index]
    healthy_numerical_feature_array = healthy_numerical_feature_df[i_col].to_numpy()
    disease_numerical_feature_array = disease_numerical_feature_df[i_col].to_numpy()

    target1_mue = np.mean(healthy_numerical_feature_array)
    target0_mue = np.mean(disease_numerical_feature_array)

    target1_std = np.std(healthy_numerical_feature_array)
    target0_std = np.std(disease_numerical_feature_array)


    means_diff  = (target0_mue-target1_mue)
    z_ppf = scipy.stats.norm.ppf(1-alpha/2)
    SE = np.sqrt(((target1_std**2)/healthy_numerical_feature_df.shape[0] + (target0_std**2)/disease_numerical_feature_df.shape[0]))
    larger_of_smaller_indicator = 'larger' if target0_mue > target1_mue else 'smaller'
    top  = means_diff  + SE*z_ppf
    bottom  = means_diff  - SE*z_ppf
    if not bottom <= 0 <= top:
        curr_string = str(i_col_index)+ ') for ' + i_col+ ' feature ' + \
        'the healthy sample mean is '+ larger_of_smaller_indicator + ' than the disease sample mean with confidence of 95%\n' 
    else:
        curr_string = str(i_col_index)+ ') for ' + i_col+ ' feature ' + \
        'Not enough confidence to reject the null hypothesis (that the healthy sample mean = disease sample mean)\n' 
    
    
    summary_string+=curr_string


print(summary_string)



##### third bullet of Q2 !

plt.figure()
sns.heatmap(numerical_feature_df.corr(), cmap='coolwarm', annot=True)
plt.title(f'Correlation matrix of all the numerical features')


corr_array = numerical_feature_df.corr().to_numpy()
np.fill_diagonal(corr_array, 0)
max_corr_row_col = np.where(corr_array == np.max(corr_array))[0]
min_corr_row_col = np.where(corr_array == np.min(corr_array))[0]


numerical_feature_columns = numerical_feature_df.columns.to_numpy()

min_cor_features = numerical_feature_columns[min_corr_row_col].tolist()
max_cor_features = numerical_feature_columns[max_corr_row_col].tolist()
string = 'The min corroletion found was between feature ' + min_cor_features[0] + ' to feature ' \
+ min_cor_features[1]
print(string)

string = 'The max corroletion found was between feature ' + max_cor_features[0] + ' to feature ' \
+ max_cor_features[1]
print(string)


ax= sns.jointplot(data = numerical_feature_df,x = min_cor_features[0], y= min_cor_features[1])
ax.fig.suptitle('joint distribution between '+min_cor_features[0]+' to ' + min_cor_features[1] + '\n maximum min correlation')

ax= sns.jointplot(data = numerical_feature_df,x = max_cor_features[0], y= max_cor_features[1])
ax.fig.suptitle('joint distribution between '+max_cor_features[0]+' to ' + max_cor_features[1] + '\n minimum min correlation')

means = [df_mean[max_cor_features[0]],df_mean[max_cor_features[1]]]
cov = np.cov(numerical_feature_df[max_cor_features[0]].to_list(), numerical_feature_df[max_cor_features[1]].to_list())
plot_2d_Gaussian_pdf(means, cov.tolist() , max_cor_features[0], max_cor_features[1])

means = [df_mean[min_cor_features[0]],df_mean[min_cor_features[1]]]
cov = np.cov(numerical_feature_df[min_cor_features[0]].to_list(), numerical_feature_df[min_cor_features[1]].to_list())
plot_2d_Gaussian_pdf(means, cov.tolist() , min_cor_features[0], min_cor_features[1])


##### fourth bullet of Q2 !

### male .vs. female

male = (categorical_feature_df['sex']==1) #male
female = ~male # female


male_numerical_feature_df = numerical_feature_df.loc[male] # seek
female_numerical_feature_df = numerical_feature_df.loc[female]

plt.figure()
sns.heatmap(female_numerical_feature_df.corr(), cmap='coolwarm', annot=True)
plt.title(f'Correlation matrix of all the female features')


plt.figure()
sns.heatmap(male_numerical_feature_df.corr(), cmap='coolwarm', annot=True)
plt.title(f'Correlation matrix of all the male features')


female_corr_array = female_numerical_feature_df.corr().to_numpy()
male_corr_array = male_numerical_feature_df.corr().to_numpy()

male_diff_famale =np.abs( male_corr_array- female_corr_array)


max_corr_row_col = np.where(male_diff_famale == np.max(male_diff_famale))[0]
min_corr_row_col = np.where(male_diff_famale == np.min(male_diff_famale))[0]


numerical_feature_columns = numerical_feature_df.columns.to_numpy()

min_cor_features = numerical_feature_columns[min_corr_row_col].tolist()
max_cor_features = numerical_feature_columns[max_corr_row_col].tolist()
string = 'The min correlation found was between feature ' + min_cor_features[0] + ' to feature ' \
+ min_cor_features[1]
print(string)

string = 'The max correlation found was between feature ' + max_cor_features[0] + ' to feature ' \
+ max_cor_features[1]
print(string)

array1 = numerical_feature_df[max_cor_features[0]].to_list()
array2 = numerical_feature_df[max_cor_features[1]].to_list()
corr, p_value = scipy.stats.pearsonr(array1, array2)


### healthy .vs. sick

healthy = (categorical_feature_df['target']==0)
disease = ~healthy

healthy_numerical_feature_df = numerical_feature_df.loc[healthy] # seek
disease_numerical_feature_df = numerical_feature_df.loc[disease]


plt.figure()
sns.heatmap(healthy_numerical_feature_df.corr(), cmap='coolwarm', annot=True)
plt.title(f'Correlation matrix of all the disease features')


plt.figure()
sns.heatmap(disease_numerical_feature_df.corr(), cmap='coolwarm', annot=True)
plt.title(f'Correlation matrix of all the healthy features')


healthy_corr_array = healthy_numerical_feature_df.corr().to_numpy()
disease_corr_array = disease_numerical_feature_df.corr().to_numpy()

healthy_diff_disease =np.abs( healthy_corr_array- disease_corr_array)


max_corr_row_col = np.where(healthy_diff_disease == np.max(healthy_diff_disease))[0]
min_corr_row_col = np.where(healthy_diff_disease == np.min(healthy_diff_disease))[0]


numerical_feature_columns = numerical_feature_df.columns.to_numpy()

min_cor_features = numerical_feature_columns[min_corr_row_col].tolist()
max_cor_features = numerical_feature_columns[max_corr_row_col].tolist()
string = 'The min correlation found was between feature ' + min_cor_features[0] + ' to feature ' \
+ min_cor_features[1]
print(string)

string = 'The max correlation found was between feature ' + max_cor_features[0] + ' to feature ' \
+ max_cor_features[1]
print(string)


array1 = numerical_feature_df[max_cor_features[0]].to_list()
array2 = numerical_feature_df[max_cor_features[1]].to_list()
corr, p_value = scipy.stats.pearsonr(array1, array2)



numerical_feature_columns_with_target = numerical_feature_columns.tolist() + ['target']

df_violin = df[numerical_feature_columns_with_target]

df_violin.loc[healthy, 'target'] = 'healthy' 
df_violin.loc[disease, 'target'] = 'disease'  



bins = np.array([20, 40, 60, 80])
df_violin['age'] = pd.cut(x=df.age, bins=bins)

for f in numerical_feature_columns_with_target:
    if f in ['target', 'age']:
        continue
    plt.figure(figsize=(15, 7))
    g = sns.violinplot(x='age', y=f, data=df_violin, hue='target',split=True, palette="flare", inner='point')
    #title = f'Age VS {f} ({desc})' if desc else f'Age VS {f}'
    plt.title('Violin plot: feature ' + f + ' V.S bins of Age')
    plt.show()
    plt.grid()
df_violin = df[numerical_feature_columns_with_target]

plt.figure()
sns.heatmap(df_violin.corr(), cmap='coolwarm', annot=True)
plt.title(f'Correlation matrix of all the disease features')

array1 = df_violin['age'].to_list()
array2 = df_violin['target'].to_list()
corr, p_value = scipy.stats.pearsonr(array1, array2)

array1 = df_violin['thalach'].to_list()
array2 = df_violin['target'].to_list()
corr, p_value = scipy.stats.pearsonr(array1, array2)


"""
Suggest, state and address at least one original question.

Our Original Question:
    we would like to know which numerical features have high and low probabilities
    to indicate whether a patient is sick or healthy.
    we used a heatmap to plot the correlation matrix with the target feature.
    from our plot we can caonclude that:
        1. the "thalach" feature is significant in positive correlation to the 
        target. we can also see that the p-value is very low (e-14). 
        this makes sense, becuase the thalach feature actually means
        "maximum heart rate acheived". the higher this value is, the more likely
        it is for the patient to have the disease.
        2. the "age" feature is negatively correlated to the target.
        we can also see it has low p value (e-5).
        this also makes sense, because the younger the patient is,
        the less likely it is for the patient to have the disease.
        
        
"""



"""
### Question 3 - Heavy Tailed Distributions and the Exponential Distribution (15 points)
### Heavy Tailed Distributions

Recall the definition of Heavy Tailed distribution from the lectures.

*A distribution is said to have a heavy right tail if its tail probabilities vanish slower than any exponential*
$$ \forall t>0, \lim_{x\to\infty} e^{tx}P(X>x)=\infty $$
Does the standard log-normal distribution have a heavy right tail? prove your answer.
"""



"""
### Special Properties of the Exponential Distribution

Let $X_1 \sim exp(\lambda_1)$ and $X_2 \sim exp(\lambda_2)$ be two independent exponential random variables.

Calculate $P(X_1 < X_2)$.
"""



"""
In this exercise you will construct trajectories of Markovian dice rolling results in the following way.<br>
The first roll, X0, is Unif(1..6)<br>
After i rolls are determined the i+1st, Xi+1, is drawn according to the row that corresponds to the value of Xi in the matrix T below. <br>
In other words, T is the transition matrix of a Markov chain and the initial distribution is uniform.

\begin{equation*}
T =
\begin{pmatrix}
0.4 & 0.2 & 0.1 & 0 & 0.1 & 0.2 \\
0.2 & 0.4 & 0.2 & 0.1 & 0 & 0.1 \\
0.1 & 0.2 & 0.4 & 0.2 & 0.1 & 0 \\
0 & 0.1 & 0.2 & 0.4 & 0.2 & 0.1 \\
0.1 & 0 & 0.1 & 0.2 & 0.4 & 0.2 \\
0.2 & 0.1 & 0 & 0.1 & 0.2 & 0.4
\end{pmatrix}
\end{equation*}
"""



"""
#### 4.A
Construct 1000 trajectories, each of length 30.
1. What do you expect the average value of all 30 numbers in a trajectory to be?
2. Compute the average value of each such trajectory. Draw a histogram of the 1000 numbers you received, using 20 bins.
3. What does the distribution look like? What are the empirical mean and the std?
"""


"""
##### 4.B
Construct 1000 trajectories, each of length 500.
1. What do you expect the average value of all 500 numbers in a trajectory to be?
2. Compute the average value of each such trajectory. Draw a histogram of the 1000 numbers you received, using 20 bins.
3. What does the distribution look like? What are the empirical mean and the std?
"""



"""
#### 4.E - Bonus (5 Points)
Let $\bar{X_n}$ be the sample average for a single trajectory of length $n$.
1. Show that **in our case**:
$$E(\bar{X_n}) =E(X_0)$$
What is it in our case ($\pi_0$ and $T$ as defined above)?
2. Show that
$$Var(\bar{X_n}) = \frac{1}{n}\sigma_0^2 + \frac{2}{n^2}\sum_{d=1}^{n-1}(n-d)Cov(X_0, X_d)$$
Calculate it for our case.
3. Formulate the CLT for Markov Chains.
4. Graphically show that the CLT holds for n=500 in our case.
"""


"""
### Question 5 - Distributions (15 Points)

Let $X$ be a random variable with a median value $Med(X) = m$. Recall that this means that $P(X\le m)=0.5$.

Consider a sample $\vec{x}(n) = x_1,...,x_n$ sampled independently from $X$. Without loss generality, assume that the observations are sorted. That is, $x_1 \le x_2 \le ... \le x_n$. Also assume that $n$ is odd and $n > 100$.

Let $R(\vec{x}(n))$ be the largest index $i \in {1,...,n}$ such that $x_i \le m$.

1. What is the distribution of $R$?
1. Given $n$, compute a function $\lambda (n)$ so that $P(x_{\lambda (n)} \le m) \ge 0.95)$ and $P(x_{\lambda (n)+1} \le m) < 0.95)$.
"""




####### THE END #########
