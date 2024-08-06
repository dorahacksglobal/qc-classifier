import numpy as np
from .feature_extraction_functions import *

#Each technique returns a list of 100 scores (for each qubit)

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


def meas_counts_qubit(binary_string):
    res = [0] * 100
    for i in range(0, len(binary_string), 100):
        for qb in range(100):
            if binary_string[i+qb] == '0':
                res[qb] += 1

    return res

def generalized_qubit_test_metric(test_function, binary_string):
    qubitStrings = [''] * 100
    for i in range(0, len(binary_string), 100):
        for qb in range(100):
            qubitStrings[qb] += str(binary_string[i+qb])

    res = [0] * 100
    for qb in range(100):
        test_metric = test_function(qubitStrings[qb])
        res[qb] = test_metric

    return res

def runs_test_qubit(binary_string):
    return generalized_qubit_test_metric(runs_test, binary_string)

def subsequences_qubit(binary_string):
    return generalized_qubit_test_metric(unique_subsequences, binary_string)

def min_entropy_qubit(binary_string):
    return generalized_qubit_test_metric(calculate_min_entropy, binary_string)

def shannon_entropy_qubit(binary_string):
    return generalized_qubit_test_metric(shannon_entropy, binary_string)

def entropy_rate_qubit(binary_string):
    return generalized_qubit_test_metric(entropy_rate, binary_string)

def permutation_entropy_qubit(binary_string):
    return generalized_qubit_test_metric(permutation_entropy, binary_string)