import pandas as pd

df = pd.read_csv('AI_2qubits_training_data copy.txt', delimiter=' ', names=['bitstring', 'label'])

squashed_string_df = df.groupby('label').apply(lambda x: ''.join(x['bitstring'].to_list()))

def window(word, size): return [word[i:i+size] for i in range(0, len(word)-size + 1)]

for i in range(len(squashed_string_df)):
    squashed_string_df.iloc[i] = window(squashed_string_df.iloc[i], 100)

squashed_string_df = pd.DataFrame(squashed_string_df.explode(), columns=['Concatenated_Data']).reset_index()
# print(squashed_string_df)
