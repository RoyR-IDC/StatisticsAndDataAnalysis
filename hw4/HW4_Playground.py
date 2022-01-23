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

    path = '/Users/royrubin/PycharmProjects/StatisticsAndDataAnalysis/hw4/raw_data_matrix_only.csv'

    dataframe = read_csv(path_to_csv=path)

    print(f'\n\n-----------\n\n')
    print(f'DF total num of nans {dataframe.isna().sum().sum()}')
    print(f'DF Info:\n{dataframe.info()}')
    print(f'DF describe:\n{dataframe.describe()}')

    dataframe.dropna(axis=0, inplace=True)

    print(f'\n\n-----------\n\n')
    print(f'DF total num of nans {dataframe.isna().sum().sum()}')
    print(f'DF Info:\n{dataframe.info()}')
    print(f'DF describe:\n{dataframe.describe()}')

    dataframe = dataframe.transpose()
    dataframe.columns = dataframe.iloc[0]
    dataframe = dataframe.drop(dataframe.index[0])

    return dataframe

def question_3_a(dataframe: pd.DataFrame):

    print(f'\n\n-----------\n\n')
    print(f'DF total num of nans {dataframe.isna().sum().sum()}')
    print(f'DF Info:\n{dataframe.info()}')
    print(f'DF describe:\n{dataframe.describe()}')

    print(f'\n\n-----------\n\n')
    value_counts = dataframe['Class'].value_counts()
    print(f'Class (label) value counts \n{value_counts}')

    print(f'\n\n-----------\n\n')

    df_20_random_features = dataframe.sample(20, axis=1)
    df_20_random_features['label'] = dataframe['Class']

    df_long = pd.melt(df_20_random_features, 'label', var_name="Gene Name", value_name="Gene Exp Value")
    df_long['Gene Exp Value'] = df_long['Gene Exp Value'].astype(float)
    sns.boxplot(y="Gene Name", x="Gene Exp Value", hue='label', data=df_long, orient="h")

    plt.show()

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
    num_of_M_samples = 49  #TODO: verify
    num_of_H_samples = 50  #TODO: verify
    total_samples = num_of_M_samples + num_of_H_samples
    probability = 1/total_samples  # it is the same for all values
    expected_value_single_sample_rank = sum([value * probability for value in range(1, total_samples+1)])
    print(f'expected_value_single_sample_rank {expected_value_single_sample_rank}')

    all_samples_rank_list = [expected_value_single_sample_rank] * num_of_M_samples
    expected_value_all_samples_rank = sum(all_samples_rank_list)
    print(f'expected_value_all_samples_rank {expected_value_all_samples_rank}')

    """
    3 b 2 
    
    """
    RS_g = expected_value_all_samples_rank
    min_RS_g_value = sum([value for value in range(1, num_of_M_samples+1)])
    max_RS_g_value = sum([value for value in range(num_of_M_samples, total_samples+1)])
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
    for current_column in dataframe.columns:
        if current_column == 'Class' or current_column == 'ID_REF':
            continue
        temp_df = dataframe[[current_column, 'Class']]
        temp_df_M_data = temp_df.loc[temp_df['Class'] == 'M']
        temp_df_H_data = temp_df.loc[temp_df['Class'] == 'H']

        M_data = temp_df_M_data[current_column].astype(float).to_numpy()
        H_data = temp_df_H_data[current_column].astype(float).to_numpy()
        statistic, pvalue = scipy.stats.ranksums(x=M_data, y=H_data)  #TODO: this function returns as a statistic not the sum but some Z score. if we want the sum, calculate it seperatly and replace the function here
        results.append(statistic)

    # after finishing, plot histogram
    sns.histplot(results)




def main():
    print("Hello World!")
    dataframe = set_up_dataframe()
    # question_3_a(dataframe)
    question_3_b(dataframe)
    print(f'Goodbye :)')

if __name__ == "__main__":
    main()
