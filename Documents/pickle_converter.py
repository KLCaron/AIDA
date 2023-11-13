import pickle
import json
import pandas as pd

# load deserialized data from pkl into a DataFrame
with open('merged_training.pkl', 'rb') as file:
    training_data = pickle.load(file)

# convert loaded data to a DataFrame
df = pd.DataFrame(training_data)

# convert dataframe to a list of dictionaries
records = df.to_dict(orient='records')

# write the list of dicitonaries to a JSON file
with open('training_data.json', 'w') as file:
    json.dump(records, file, indent=2)
