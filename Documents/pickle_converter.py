import pickle
import pandas as pd

# load deserialized data from pkl into a DataFrame
with open('merged_training.pkl', 'rb') as file:
    training_data = pickle.load(file)

# convert loaded data to a DataFrame
df = pd.DataFrame(training_data)

# serialize DataFrame to JSON
df.to_json('training_data.json', orient='records', lines=True)
