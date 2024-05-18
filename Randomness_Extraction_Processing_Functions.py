from math import log2
from scipy.fft import fft, ifft
from scipy.special import erfc
import numpy as np

def calculate_2bit_shannon_entropy(binary_string):
    # Ensure the string length is a multiple of 2 for exact 2-bit grouping
    if len(binary_string) % 2 != 0:
        raise ValueError("Binary string length must be a multiple of 2.")
    
    # Define possible 2-bit combinations
    #patterns = ['0000', '1000', '1100', '1110', '1111', '0100', '0110', '0111', '0010', '0011', '0001', '1001', '1101', '0110', '0101', '1010']
    patterns = ['00', '10', '11', '01']
    frequency = {pattern: 0 for pattern in patterns}
    
    # Count frequency of each pattern
    for i in range(0, len(binary_string), 2):
        segment = binary_string[i:i+2]
        if segment in patterns:
            frequency[segment] += 1
    
    # Calculate total segments counted
    total_segments = sum(frequency.values())
    
    # Calculate probabilities and entropy
    entropy = 0
    for count in frequency.values():
        if count > 0:
            probability = count / total_segments
            entropy -= probability * log2(probability)
    
    return entropy


def calculate_min_entropy(binary_string):
    numZeros = 0

    #find max 1s or 0s
    for i in range(0, len(binary_string)):
        if binary_string[i] == '0':
            numZeros += 1
    maxDigit = max(numZeros, (len(binary_string)-numZeros))

    #calculate min entropy
    prob = maxDigit/len(binary_string)
    return -(log2(prob))


def classic_spectral_test(bit_string):
    """
    Perform the classic spectral test using Discrete Fourier Transform (DFT)
    on the input bit string.

    Args:
        bit_string (str): A string of 0s and 1s representing the bit sequence.

    Returns:
        float: The P-value of the test.
    """
    # Convert bit string to numpy array of -1 and 1
    bit_array = 2 * np.array([int(bit) for bit in bit_string]) - 1

    # Compute the DFT of the bit array
    dft = fft(bit_array)

    # Calculate the number of bits for the first half of the DFT output
    n_half = len(bit_string) // 2 + 1

    # Compute the modulus of the first half of the DFT output
    mod_dft = np.abs(dft[:n_half])

    # Compute the 95% peak height threshold
    threshold = np.sqrt(np.log(1 / 0.05) / len(bit_string))

    # Count the number of peaks below the threshold
    peaks_below_threshold = np.sum(mod_dft < threshold)

    # Compute the expected number of peaks
    expected_peaks = 0.95 * n_half

    # Compute the test statistic
    d = (peaks_below_threshold - expected_peaks) / np.sqrt(len(bit_string) * 0.95 * 0.05)

    # Compute the P-value using the complementary error function
    p_value = erfc(np.abs(d) / np.sqrt(2)) / 2

    return d