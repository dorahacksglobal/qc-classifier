import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import zlib
from pydotplus import graph_from_dot_data
from IPython.display import Image
from sklearn.tree import export_graphviz
from Randomness_Extraction_Processing_Functions import *

# File path
file_path = '/Users/sid/Code/QRNGclassifier/AI_2qubits_training_data.txt'


def readData(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                binary_number, label = line.strip().split()
                data.append((binary_number, int(label)))

    # Convert the data into a DataFrame
    return pd.DataFrame(data, columns=['binary_number', 'label'])



# concatenate optimal number of input rows for highest accuracy (pros and cons)
def concatenateData(df, num_concats):
    new_df = pd.DataFrame({'Concatenated_Data': [''] * (len(df) // num_concats), 'label': [''] * (len(df) // num_concats)})

    # Loop through each group of 10 rows and concatenate their 'Data' strings
    for i in range(0, len(df), num_concats):
        new_df.iloc[i // num_concats, 0] = ''.join(df['binary_number'][i:i+num_concats])
        new_df.iloc[i // num_concats, 1] = df['label'][i]
    return new_df


def processData():
    df = readData('/Users/sid/Code/QRNGclassifier/AI_2qubits_training_data.txt')
    new_df = concatenateData(df, 40)

    #new_df['spectral_randomness'] = new_df['Concatenated_Data'].apply(classic_spectral_test)
    new_df['shannon_entropy'] = new_df['Concatenated_Data'].apply(calculate_2bit_shannon_entropy)
    new_df['min_entropy'] = new_df['Concatenated_Data'].apply(calculate_min_entropy)

    # Preprocess the binary_number column to convert each bit to a separate feature column
    df_features = pd.DataFrame(new_df['Concatenated_Data'].apply(list).tolist())
    new_df = pd.concat([new_df.drop(columns='Concatenated_Data'), df_features], axis=1)
    return new_df



'''def compression_complexity(data):
    # Convert binary string data to bytes if it's not already in byte form
    if isinstance(data, str):
        data = data.encode('utf-8')

    compressed_data = zlib.compress(data)
    compressed_length = len(compressed_data)
    return compressed_length
df['compression_complexity'] = df['binary_number'].apply(compression_complexity)
'''

new_df = processData()
print(new_df.tail(10))

# Split the data into features (X) and labels (y)
X = new_df.drop(columns='label').values
#print(X)
y = new_df['label'].values
y=y.astype('int')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print(X_test, y_test)

def gradient_boosting():
    # Create the Gradient Boosting classifier
    gb_model = GradientBoostingClassifier(random_state=42)

    # Train the model
    gb_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_gb = gb_model.predict(X_test)

    sub_tree_1 = gb_model.estimators_[1, 0]

    # Calculate the accuracy of the Gradient Boosting model
    accuracy_gb = accuracy_score(y_test, y_pred_gb)

    print("Gradient Boosting Accuracy:", accuracy_gb)
    
    #printing visualization of sample decision tree in gradient booster
    '''
    dot_data = export_graphviz(
        sub_tree_1,
        out_file=None, filled=True, rounded=True,
        special_characters=True,
        proportion=False, impurity=False, # enable them if you want
    )
    graph = graph_from_dot_data(dot_data)
    png = graph.create_png()
    from pathlib import Path
    Path('./out.png').write_bytes(png)'''

def gradient_boosting_grid_search():
    gb_model = GradientBoostingClassifier(random_state=42)

    # Define the hyperparameter grid for Grid Search
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
    }

    # Perform Grid Search with cross-validation (cv=5) to find the best hyperparameters
    grid_search = GridSearchCV(gb_model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and model
    best_model = grid_search.best_estimator_

    # Make predictions on the test set using the best model
    y_pred_gb = best_model.predict(X_test)

    # Calculate the accuracy of the Gradient Boosting model with the best hyperparameters
    accuracy_gb = accuracy_score(y_test, y_pred_gb)
    print("Gradient Boosting Accuracy:", accuracy_gb)

gradient_boosting()