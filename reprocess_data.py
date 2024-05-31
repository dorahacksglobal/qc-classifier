import pandas as pd

df = pd.read_csv('AI_2qubits_training_data.txt', delimiter=' ', names=['bitstring', 'label'])

squashed_string_df = df.groupby('label').apply(lambda x: ''.join(x['bitstring'].to_list()))

def window(word, size=1, gap=1): return [word[i:i+size] for i in range(0, len(word)-size + 1, gap)]

for i in range(len(squashed_string_df)):
    squashed_string_df.iloc[i] = window(squashed_string_df.iloc[i], 100, 2)  # sample 2 qubits at a time

# TODO: assume bits are processed in a single stream - k-fold cross validation should split into k chunks ahead of time to preserve time dependency information?
squashed_string_df = pd.DataFrame(squashed_string_df.explode(), columns=['Concatenated_Data']).reset_index()
# print(squashed_string_df)

# squashed_string_df.to_csv('AI_2qubits_training_data_windowed.txt', sep=' ', names=['bitstring', 'label'])

# can save this into a file to cache the result
