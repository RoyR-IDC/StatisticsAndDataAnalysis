import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as stats
import seaborn as sns
from scipy.special import comb

from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def convert_to_numeric(c):
    try:
        return pd.to_numeric(c)
    except:
        return c

def pre_processing_csv_file(csv_path):
    first_actual_row = 59  # all previous rows are some irrelevant metadata
    df = pd.read_csv(csv_path, encoding='ISO-8859-1', header=None)
    temp_df = df.drop(df.index[:first_actual_row]).reset_index(drop=True)
    temp_df.dropna(how='any', axis=0)
    df = temp_df.T  # make patients as rows and [Class, IDs, and genes] as columns, for better readability
    column_names = df.iloc[0]  # get column names
    df = df[1:]  # don't include headers row in data
    df.columns = column_names
    df = df.apply(convert_to_numeric)
    df_dtypes = df.dtypes
    return df_dtypes, df
def generate_pair_box_plot():
    BINS = 10
    N_genes = 2
    # starting from index 2 which is the first gene (0 is "Class" and 1 is "ID")
    random_indices = np.random.randint(low=2, high=df.shape[1], size=N_genes)
    for ax_ind, i in enumerate(random_indices):
        plt.figure()
        curr_columns = columns_list[i]
        ax1 = sns.boxplot(x="Class", y=curr_columns, data=df)
        ax1 = sns.swarmplot(x="Class", y=curr_columns, data=df, color=".25")

        gene_name = curr_columns
        ax1.set_title(f'Pair Boxplot of Gene: {gene_name}\n  **Myocardial Infraction** vs **Healthy** of patients')
    return
def calculate_RS_for_each_gene(g):
    # get ranks of all samples for this gene g
    ranks = g.rank()
    # sum of the ranks of ONLY the 'M' samples for this gene g
    ranks_sums = ranks[M_samples].sum()
    return  ranks_sums
def calculate_t_test(g):
    '''Calculate the T-test for the means of two independent samples of scores'''
    # this is the one Zohar mentioned in class
    return stats.ttest_ind(g[M_samples], g[H_samples], equal_var=True)
def calculate_WRS(g):
    return stats.ranksums(g[M_samples], g[H_samples])
"""
pre-processing data 
"""
csv_path = r"C:\Msc\Git\StatisticsAndDataAnalysis\hw4\AMI_GSE66360_series_matrix.csv"

df_dtypes, df = pre_processing_csv_file(csv_path)

print(df_dtypes)
print(df)

"""
1.a How many genes profiled
"""
# the amount of columns - the two first columns are Class & ID_REF
amount_of_columns = df.shape[1]
curr_string = 'The amount of genes profiled is ' + str(amount_of_columns-2)

"""
1.b How many samples (subjects/patients) in total
"""

# the amount of rows - the two first columns are Class & ID_REF
amount_of_rows = df.shape[0]
curr_string = 'The amount of samples (subjects/patients) in total is ' + str(amount_of_rows)
print(curr_string)
"""
1.c How many samples in each class?
"""
hist = df['Class'].hist()
plt.title('Histogram of amount class')
count_df = df['Class'].value_counts()
print(count_df)


"""
If there are missing values, then remove the entire row
(gene) from the data matrix.
How many rows left now?
"""
# check if there is null\missing values in data frame

df.dropna(axis='columns', inplace=True) # Removing columns with missing values
amount_of_miss_rows = amount_of_columns-df.shape[1]

amount_of_columns = df.shape[1]
columns_list = df.columns.to_list()
curr_string = 'The amount of removed columns is ' + str(amount_of_miss_rows) + \
            '\n The amount of left columns is ' + str(amount_of_columns)
    
print(curr_string)
print(df)



"""
1.5) Pick 20 genes at random. Draw 20 pair boxplots in one
figure comparing expression levels of each of these genes
in the two classes M and H.
"""
mask = df['Class'] == 'H'
df_H = df[mask]
df_M = df[~mask]

generate_pair_box_plot()

"""
WRS for differential expression (DE)
Consider some gene, g. Under the null model (which assumes that
for g there is no M vs H DE), what is the expected sum of ranks of
g’s expression levels measured for samples labeled M?


### b) WRS for differential expression (DE)
1.b 
Consider some gene, g. Under the null model (which assumes that for g there is no M vs H DE),
what is the expected sum of ranks of g’s expression levels measured for samples labeled M?
"""

H_samples = df.loc[df['Class'] == 'H']
M_samples = df.loc[df['Class'] == 'M']

H_samples_count, M_samples_count =  count_df.to_list()
B = H_samples_count + M_samples_count
excepted_mue_T = ((M_samples_count)*(B+1))/(2)
curr_string = 'The Excepted value is ' + str(excepted_mue_T)

"""
2.b  
Denote this sum of ranks by RS(g). What is the maximal value, c,
that RS(g) can take?
"""
"""
Answer:
    note that WRS is not care abount values because of the ranks
    therefore if the data of M,H are seprated, and farthere more the M are all
    smaller than the H values
"""

maximum_R_s = np.sum(np.arange(H_samples_count, B)+1)

"""
3.b 
Under the null model, what is the probability of RS(g) = c?
(Provide a formula for this and explain it)
"""
"""
Answer:
    note that this is combination order problem, however there is only one option
    to order the value of M\H
    
"""

denominator = comb(B, M_samples_count, exact=True)
probability = 1 / denominator

"""
4.b 
Under the null model, what is the probability of RS(g) = c-1? what is
the probability of RS(g) = c-2?
(Provide formulas and explain them)
"""





"""
5.b
Draw a histogram of the values of RS(g) in the dataset. Here g
ranges over all genes in the data (after the clean-up). Compute the
IQR for this distribution and present it on the plot with the histogram

"""
"""
5. Draw a histogram of the values of RS(g) in the dataset. Here g ranges over all genes in the data (after the clean-up). Compute the IQR for this distribtuion.
"""



H_samples = df['Class'] == 'H'
M_samples = df['Class'] == 'M'


def generate_RS_histogram(df):
    global iqr
    # generate RS value for each gene
    ranked_df = df.iloc[:, 2:].apply(calculate_RS_for_each_gene, axis=0)
    ranked_df.hist(bins=30, density=True)
    # calculate precentile
    q75, q25 = np.percentile(ranked_df, [75, 25])
    # calculate IQR
    iqr = q75 - q25
    plt.title('histogram RS(g) vs samples labeled M')
    plt.ylabel('Density')
    plt.xlabel('RS(g)')
    max_y = plt.gca().get_ylim()[1]
    min_x = plt.gca().get_xlim()[0]
    plt.text(min_x * 1.05, max_y * 0.9, f'IQR: {iqr}', color='g', size=15, bbox=dict(facecolor='y', alpha=0.7))
    plt.show()


generate_RS_histogram()

"""
***
### c)
Differential Expression
The purpose is to determine the statistical significance of
differential expression (DE) observed for each gene in H vs M.
Evaluate the DE in both one-sided directions for every gene,
using both Student t-test and WRS test.
Report the number of genes overexpressed in M vs H (M > H) at
a p-value better (≤) than 0.07 and separately genes
underexpressed in M vs H (M < H) at a p-value better than 0.07.
For both directions use both a Student t-test and a WRS test.

"""

alpha = 0.07 # certainty of 93%



# starting from index 2 which is the first gene (0 is "Class" and 1 is "ID")
TTEST_df = df.iloc[:,2:].apply(calculate_t_test, axis=0, result_type='expand')
TTEST_df.index=['statistic', 'p-value']

print(TTEST_df)


TTEST_values = TTEST_df.T['statistic'] # The calculated t-statistic
TTEST_p_values = TTEST_df.T['p-value'] / 2  # Since "ttest_ind()" returns the 2-tailed p-value of the test, we divide the values by 2 ("...a two-sided test for the null hypothesis")

TTEST_overexpressed_genes_df = TTEST_df.T[(TTEST_values > 0) & (TTEST_p_values < alpha)]
TTEST_underexpressed_genes_df = TTEST_df.T[(TTEST_values <= 0) & (TTEST_p_values < alpha)]

print(f'TTEST overexpressed genes in M vs H is {TTEST_overexpressed_genes_df.shape[0]}') # shape[0] for the number of rows left after filtering
print(f'TTEST underexpressed genes in M vs H is {TTEST_underexpressed_genes_df.shape[0]}')


# starting from index 2 which is the first gene (0 is "Class" and 1 is "ID")
WRS_df = df.iloc[:,2:].apply(calculate_WRS, axis=0, result_type='expand')
WRS_df.index=['statistic', 'p-value']

print(WRS_df)

WRS_df_copy = WRS_df.copy()

WRS_values = WRS_df_copy.T['statistic'] # The test statistic under the large-sample approximation that the rank sum statistic is normally distributed
WRS_p_values = WRS_df_copy.T['p-value'] / 2  # Since "ranksums()" returns the 2-sided p-value of the test, we divide the values by 2

WRS_overexpressed_genes_df = WRS_df_copy.T[(WRS_values > 0) & (WRS_p_values < alpha)]
WRS_underexpressed_genes_df = WRS_df_copy.T[(WRS_values <= 0) & (WRS_p_values < alpha)]

print(f'WRS overexpressed genes in M vs H is {WRS_overexpressed_genes_df.shape[0]}') # shape[0] for the number of rows left after filtering
print(f'WRS underexpressed genes in M vs H is {WRS_underexpressed_genes_df.shape[0]}')


"""
Compute Kendall tho correlations in all pairs within D (160
choose 2 numbers). Represent the correlation matrix as a 160x160
heatmap.
"""

"""
NOTE that the genes with smaller p values are more significant
we will choose the 80 most significant (smallest) genes from each one of the one-sided WRS DE 
"""
n = 80
over_significant_df = WRS_overexpressed_genes_df.sort_values(by='p-value').head(n) # sorting by the p-values
under_significant_df = WRS_underexpressed_genes_df.sort_values(by='p-value').head(n)

# Generate a set of 160 genes, D, which is the union of the above two sets
most_significant_df  = pd.concat([over_significant_df, under_significant_df])
most_significant_df = df[most_significant_df.index]

def gen_kendall_pairs(df):

    corrs: Dict[Tuple[str, str], 
                Tuple[np.float64, np.float64]] = dict()

    for gene1 in df.columns:
        for gene2 in df.columns:
            # avoiding correlations of a gene with itself
            if gene1 == gene2:
                continue

            key = (gene1, gene2)
            key_opposite = (gene2, gene1)
            # avoiding duplicates since tau(g1, g2) == tau(g2, g1)
            if key not in corrs and key_opposite not in corrs: 
                # return stats.kendalltau  --> tau, p-value
                kndl = stats.kendalltau(df[gene1], df[gene2])
                # update
                corrs[key] = (kndl[0], kndl[1]) 
    
    return corrs

corrs = gen_kendall_pairs(most_significant_df)

D_kendall_correlations_df = most_significant_df.corr(method='kendall') 
plt.figure(figsize = (25,15))
sns.heatmap(D_kendall_correlations_df, annot=True,cmap="YlGnBu", vmin=-1, vmax=1)

plt.title('160 Most Significant Genes - Correlation', fontsize=20)
plt.xlabel('gene name', fontsize=15)
plt.ylabel('gene name', fontsize=15)
plt.show()


"""
Under a NULL model that assumes that genes are pairwise
independent, what is the expected value for tho ?
"""








































