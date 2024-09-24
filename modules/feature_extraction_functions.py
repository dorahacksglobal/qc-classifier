import numpy as np
from math import log2, sqrt
import itertools
from scipy.fft import fft
from scipy.special import erfc
import itertools
from scipy.stats._entropy import entropy

def shannon_entropy(input_string: str, possible_values: str, chunk_size: int = 1):
    """Calculate Shannon (standard) entropy for a given string. Uses scipy implementation

    Args:
        input_string (string): data, in string format, to extract entropy calculation from (required)
        possible_values (string): every possible value present in the input data (required)
        chunk_size (int): the size of the chunks to calculate entropy on; must be divisible by size of input data string and also be <= 12 (optional, defaults to 1)

    Returns:
        float: Shannon entropy calculation

    """
    
    input_length = len(input_string)

    if (input_length % chunk_size) != 0:
        return ValueError('The chunk size must be a factor of the length of the input string; i.e. input_length mod (chunck_size) = 0')
    if chunk_size > 12:
        return ValueError('The supplied chunk size must be less than 13 for space constraints')

    patterns = [''.join(pattern) for pattern in itertools.product(possible_values, repeat=chunk_size)]

    frequencies = {key: 0 for key in patterns}
    
    for i in range(0, len(input_string), chunk_size):
        segment = input_string[i:i+chunk_size]
        if segment not in patterns:
            return ValueError('The input string contains values not present in the possible_values argument')
        else:
            frequencies[segment] += 1
    
    total_segments = input_length / chunk_size

    probabilities = []
    for pattern in frequencies:
        probabilities.append(frequencies[pattern] / total_segments)

    return entropy(probabilities, base=2)
    

def shannon_entropy_integer(input_string: str, chunk_size: int = 1):
    """Calculate Shannon (standard) entropy for a given string, but by converting each chunk to an integer first, avoiding the cartesian product 
       complexity for all possible bitstrings. Chunk size however must be less than 30 for practicality. Uses scipy implementation

    Args:
        input_string (string): data, in string format, to extract entropy calculation from (required)
        possible_values (string): every possible value present in the input data (required)
        chunk_size (int): the size of the chunks to calculate entropy on; must be divisible by size of input data string and < 30 (optional, defaults to 1)

    Returns:
        float: Shannon entropy calculation

    """

    input_length = len(input_string)

    if chunk_size > 29:
        return ValueError('The supplied chunk size must be less than 30 for space constraints')
    if (input_length % chunk_size) != 0:
        return ValueError('The chunk size must be a factor of the length of the input string; i.e. input_length mod (chunck_size) = 0')
    
    #Array size is total possible binary sequences (converted to integers)
    frequencies = [0] * (2**chunk_size)
    #iterate though each "chunk"
    for i in range(0, len(input_string), chunk_size):
        binarySum = 0
        for j in range(i, i+chunk_size):
            if input_string[j] == '1':
                binarySum += 2 ** (i+chunk_size-j-1)
        frequencies[binarySum] += 1
    
    total_segments = input_length / chunk_size
    probabilities = []
    for i in frequencies:
        probabilities.append(i / total_segments)

    return entropy(probabilities, base=2)
 

def runs_test(bit_string):
    n = len(bit_string)
    runs = 1  # Start with the first run
    for i in range(1, n):
        if bit_string[i] != bit_string[i - 1]:
            runs += 1
    
    return runs

def longest_run(bit_string):
    cur = 1
    longest = 1
    for i in range(1, len(bit_string)):
        if bit_string[i] == bit_string[i-1]:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 1
    return longest


def cumulative_sums_test(bit_string):
    bit_array = np.array([int(bit) for bit in bit_string])
    adjusted = 2 * bit_array - 1
    cumulative_sum = np.cumsum(adjusted)
    max_excursion = np.max(np.abs(cumulative_sum))
    return max_excursion


def unique_subsequences(bit_string, length=4):
    bit_array = np.array([int(bit) for bit in bit_string])
    n = len(bit_array)
    subsequences = set()
    
    for i in range(n - length + 1):
        subseq = tuple(bit_array[i:i+length])
        subsequences.add(subseq)
    
    return len(subsequences)


# Calculate min-entropy
def calculate_min_entropy(sequence):
    sequence = np.asarray(sequence, dtype=float)  # Convert sequence to float
    p = np.mean(sequence)  # Proportion of ones
    max_prob = max(p, 1 - p)
    if max_prob == 0:  # Handle the case where all bits are the same
        return 0
    min_entropy = -np.log2(max_prob)
    return min_entropy


def markov_chain_transition_counts(bit_string):
    count = 0
    for i in range(1, len(bit_string)):
        if bit_string[i] != bit_string[i-1]:
            count += 1
    return count