import numpy as np
import pandas as pd
from .feature_extraction_functions import *
from .qubit_feature_extraction_features import * 


def load_data_into_df(file_path, label_names, label_filtering):
    """Load data and labels from a space seperated txt dataset file into a Pandas dataframe.

    Args:
        file_path (string): the path to the dataset file (ensure proper scope)
        label_names (array-like): the label names for the dataframe column headers label_filtering: not yet implemented (TO-DO)

    Returns: 
        DataFrame: Pandas DF containing input data and labels

    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            entries = line.strip().split()
            if len(label_names) != len(entries[1:]):
                return 'ERROR: number of label names do not match number of labels in training dataset for at least one line'
            data.append(entries)
    
    columns = ['binary_number'] + label_names
    df = pd.DataFrame(data, columns=columns)
    return df


def filter_by_label(df: pd.DataFrame, labels: list[str]):
    """Returns subset of inputted DataFrame containing only specified labels.

    Args:
        df (DataFrame): Dataframe containing data and labels to be filtered
        labels (array-like): labels to keep in subset df (i.e. which labels not to filter out)

    Returns: 
        DataFrame: Pandas DF containing subset of labels

    """
    for label in labels:
        if label not in df.columns:
            raise NameError('label not in dataset')
    return df[labels]


def concatenate_data(df: pd.DataFrame, num_concats: int):
    """Concatenates binary strings for input training data (i.e. combines successive QRNG binary string data points to make longer binary strings)

    Args: 
        df (DataFrame): DataFrame containing the data to concatenate
        num_concats (int): how many successive lines to concatenate together at a time (i.e. for initial string size = 100 and num_concats = 500, 
                           binary strings in the new df would have length = 500)

    Returns:
        DataFrame: new df with concatenated data (length of returned df would be smaller than input df by a factor of num_concats)

    """
    if df.shape[0] % num_concats != 0:
        return 'ERROR: number of concats must be a divisor of the length of the dataframe'
    label_names = df.columns[1:]
    concatenated_df = pd.DataFrame({
        'Concatenated_Data': [''] * (len(df) // num_concats), 
    })
    for count, label in enumerate(label_names):
        concatenated_df.insert(count+1, column=label, value=([''] * (len(df) // num_concats)))

    # Loop through each group of num_concats rows and concatenate their 'binary_number' strings
    for i in range(0, len(df), num_concats):
        concatenated_df.iloc[i // num_concats, 0] = ''.join(df['binary_number'][i:i + num_concats])
        for count, label in enumerate(label_names):
            concatenated_df.iloc[i // num_concats, count+1] = df[label][i]

    return concatenated_df


def apply_feature_extraction_functions(df: pd.DataFrame, functions_to_apply: dict):
    test_functions = {
        'autocorrelation': autocorrelation_test,
        'cumulative_sums': cumulative_sums_test,
        'spectral_test': classic_spectral_test,
        'frequency_test': frequency_test,
        'runs_test': runs_test,
        'shannon_entropy': shannon_entropy,
        'min_entropy': calculate_min_entropy,
        'entropy_rate': entropy_rate,
        'lyapunov_exponent': lyapunov_exponent,
        'permutation_entropy': permutation_entropy,
        'sample_entropy': sample_entropy,
        'unique_subsequences': unique_subsequences,
        'random_excursions_test': random_excursions_test,
        'longest_run_ones_test': longest_run_ones_test,
        'cumulative_sums_test': cumulative_sums_test,
        'binary_matrix_rank_test': binary_matrix_rank_test,
        'maurer_universal_test': maurer_universal_test,
        'linear_complexity': linear_complexity,
    }

    for test in functions_to_apply:
        if test not in test_functions:
            raise ValueError(f"Invalid randomness test: {test}")
        df[test] = df['Concatenated_Data'].apply(test_functions[test])

    df.dropna(axis=1)

    return df


def make_bit_features(df: pd.DataFrame):
        df_features = pd.DataFrame(df['Concatenated_Data'].apply(list).tolist(), dtype=float)
        processed_df = pd.concat([df.drop(columns='Concatenated_Data'), df_features], axis=1)
        return processed_df


def remove_bitstring_as_feature(df: pd.DataFrame):
    df.drop(columns=df.columns[0], inplace=True)
    return df


def apply_individual_qubit_functions(df: pd.DataFrame, functions: list, num_qubits: int):

    test_functions = {
        'counts': meas_counts_qubit,
        'runs': runs_test_qubit,
        'unq_subsq':  subsequences_qubit,
        'min_entro': min_entropy_qubit,
        'shan_entro': shannon_entropy_qubit,
        'entro_rate': entropy_rate_qubit,
        'perm_entro': permutation_entropy_qubit,
        'longest_run': longest_run_qubit,
        'markov_counts': markov_counts_qubit
    }

    for function in functions:
        if function not in test_functions:
            raise ValueError(f"Invalid randomness test: {function}")
        
    for function in functions:
        results = []
        for index, row in df.iterrows():
            string = row['Concatenated_Data']
            res = test_functions[function](string, num_qubits)
            results.append(res)
        
        for qb in range(num_qubits):
            qubitRes = [results[i][qb] for i in range(len(results))]
            df[f'{function}_qb_{qb}'] = qubitRes
    
    return df


def add_PRNG_data(file_path, qubits, numLines):
    rng = np.random.default_rng()
    with open(file_path, 'a') as file:
        for i in range(numLines):
            prng_data = rng.integers(2, size=qubits)
            stringPRNG = ''.join([str(i) for i in prng_data])
            output = stringPRNG + ' non-quantum N/A\n'
            file.write(output)
    file.close()

