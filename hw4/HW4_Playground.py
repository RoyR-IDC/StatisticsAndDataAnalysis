###---- HW 4 ----###
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib import pyplot as plt


def read_csv(path_to_csv: str) -> pd.DataFrame:
    try:
        dataframe = pd.read_csv(path_to_csv, dtype=object, header=None, index_col=None)
    except:
        raise AssertionError(f'Could not open csv file')

    return dataframe


def set_up_dataframe():
    # define path
    path = '/Users/royrubin/PycharmProjects/StatisticsAndDataAnalysis/hw4/raw_data_gene_matrix_only.csv'
    # read file
    dataframe = read_csv(path_to_csv=path)
    # update file format
    dataframe = dataframe.transpose()
    dataframe.columns = dataframe.iloc[0]
    dataframe = dataframe.drop(dataframe.index[0])
    # return file
    return dataframe


def question_3_a(dataframe: pd.DataFrame):
    """
    3 A 1
    1) How many genes profiled?
    """
    original_num_of_genes = dataframe.shape[1]
    print(f'Number of genes profiled: {original_num_of_genes}')

    """
    3 A 2
    2) How many samples (subjects/patients) in total?
    """
    total_num_of_samples = dataframe.shape[0]
    print(f'Number of samples profiled: {total_num_of_samples}')

    """
    3 A 3
    3) How many samples in each class?
    """
    value_counts = dataframe['Class'].value_counts()
    print(f'Number of samples in each class \n{value_counts}')

    """
    3 A 4

    4) If there are missing values, then remove the entire row (gene) from the data matrix. How many rows left now?
    """
    original_num_of_nans = dataframe.isna().sum().sum()
    dataframe.dropna(axis=1, inplace=True)  # because the DF is transposed, this is on the columns
    new_num_of_nans = dataframe.isna().sum().sum()
    assert original_num_of_nans != 0 and new_num_of_nans == 0
    new_num_of_genes = dataframe.shape[1]
    print(f'Number of genes left after deleting empty ones: {new_num_of_genes}'
          f'\nNumber of genes deleted: {original_num_of_genes - new_num_of_genes}')

    """
    3 A 5

    5) Pick 20 genes at random. Draw 20 pair boxplots in one figure comparing 
    expression levels of each of these genes in the two classes M and H.
    """

    # df_20_random_features = dataframe.sample(20, axis=1)
    # df_20_random_features['label'] = dataframe['Class']
    #
    # df_long = pd.melt(df_20_random_features, 'label', var_name="Gene Name", value_name="Gene Exp Value")
    # df_long['Gene Exp Value'] = df_long['Gene Exp Value'].astype(float)
    #
    # plt.figure(0)
    # sns.boxplot(y="Gene Name", x="Gene Exp Value", hue='label', data=df_long, orient="h")
    # plt.show()


def question_3_b(dataframe: pd.DataFrame):
    """
    3 b 1
    Consider some gene, g. Under the null model (which assumes that for g there is no M vs H DE),
    what is the expected sum of ranks of g’s expression levels measured for samples labeled M?

    Answer:

    first of all, not that there are 49 * M samples, 50 * H samples
    E(single rank) = sum(val*P(val)) = 1 * 1/(49+50) + 2 * 1/(49+50) + ... + (49+50) * 1/(49+50)

    E(sum of all ranks) = sum(E(single rank)) = sum()
                                        equiprobable
    """
    num_of_M_samples = 49  # TODO: verify
    num_of_H_samples = 50  # TODO: verify
    total_samples = num_of_M_samples + num_of_H_samples
    probability = 1 / total_samples  # it is the same for all values
    expected_value_single_sample_rank = sum([value * probability for value in range(1, total_samples + 1)])
    print(f'expected_value_single_sample_rank {expected_value_single_sample_rank}')

    all_samples_rank_list = [expected_value_single_sample_rank] * num_of_M_samples
    expected_value_all_samples_rank = sum(all_samples_rank_list)
    print(f'expected_value_all_samples_rank {expected_value_all_samples_rank}')

    """
    3 b 2 

    """
    RS_g = expected_value_all_samples_rank
    min_RS_g_value = sum([value for value in range(1, num_of_M_samples + 1)])
    max_RS_g_value = sum([value for value in range(num_of_M_samples, total_samples + 1)])
    print(f'min_RS_g_value {min_RS_g_value} max_RS_g_value {max_RS_g_value}')

    """
    3 b 3

    if c = max_value for RS_g,
    then P(RS_g == c) = 1 / (number of rank sum options)
    """

    """
    3 b 4 
    TODO:
    """

    """
    3 b 5
    """
    results = []
    for index, current_column in enumerate(dataframe.columns):
        if current_column == 'Class' or current_column == 'ID_REF':
            # In this case, we dont need to calculate the sum of ranks
            continue

        # print just for progress tracking
        if index % 500 == 0:
            print(f'Progress: now at index {index}')

        # First, keep only relevant columns from original dataframe
        temp_df = dataframe[[current_column, 'Class']]

        # Sort and rank the dataframe
        sorted_df = temp_df.sort_values(current_column, ascending=False)
        sorted_df['ranks'] = range(1, len(sorted_df) + 1)
        final_df = sorted_df.drop(columns=[current_column])

        # Get sum of ranks
        sum_of_ranks_both_classes = final_df.groupby(['Class']).sum()
        current_RS_g = sum_of_ranks_both_classes.loc['M'].item()

        # Save current result
        results.append(current_RS_g)

    # after finishing, compute IQR, then plot histogram (with iqr line
    distribution_iqr = scipy.stats.iqr(results)

    # plot the figure - histogram with an iqr line
    plt.figure(0)
    sns.histplot(results)
    plt.axhline(y=distribution_iqr, color='r', linestyle='-')
    plt.show()

    print(f'Finished Q3B')


def main():
    print("Hello World!")
    dataframe = set_up_dataframe()
    question_3_a(dataframe)
    question_3_b(dataframe)
    print(f'Goodbye :)')


if __name__ == "__main__":
    main()

