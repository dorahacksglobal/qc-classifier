import numpy as np
from .feature_extraction_functions import *

#Each technique returns a list of 100 scores (for each qubit)

def meas_counts_qubit(binary_string, num_qubits):
    res = [0] * num_qubits
    for i in range(0, len(binary_string), num_qubits):
        for qb in range(num_qubits):
            if binary_string[i+qb] == '0':
                res[qb] += 1

    return res

def generalized_qubit_test_metric(test_function, binary_string, num_qubits):
    qubitStrings = [''] * num_qubits
    for i in range(0, len(binary_string), num_qubits):
        for qb in range(num_qubits):
            qubitStrings[qb] += str(binary_string[i+qb])

    res = [0] * num_qubits
    for qb in range(num_qubits):
        test_metric = test_function(qubitStrings[qb])
        res[qb] = test_metric

    return res

def runs_test_qubit(binary_string, num_qubits):
    return generalized_qubit_test_metric(runs_test, binary_string, num_qubits)

def subsequences_qubit(binary_string, num_qubits):
    return generalized_qubit_test_metric(unique_subsequences, binary_string, num_qubits)

def min_entropy_qubit(binary_string, num_qubits):
    return generalized_qubit_test_metric(calculate_min_entropy, binary_string, num_qubits)

def shannon_entropy_qubit(binary_string, num_qubits):
    return generalized_qubit_test_metric(shannon_entropy, binary_string, num_qubits)

def entropy_rate_qubit(binary_string, num_qubits):
    return generalized_qubit_test_metric(entropy_rate, binary_string, num_qubits)

def permutation_entropy_qubit(binary_string, num_qubits):
    return generalized_qubit_test_metric(permutation_entropy, binary_string, num_qubits)

def longest_run_qubit(binary_string, num_qubits):
    return generalized_qubit_test_metric(longest_run, binary_string, num_qubits)

def markov_counts_qubit(binary_string, num_qubits):
    return generalized_qubit_test_metric(markov_chain_transition_counts, binary_string, num_qubits)