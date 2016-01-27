#Imports
import pandas as pd
import json

####################################################
# Initial Data Ingest
####################################################

# Using what we have learned about pandas, we break
# the initial Yelp Academic dataset into 3 separate
# files, all containing JSON objects for users,
# reviews and businesses, respectively. We take the
# paths where these .json files are located and read 
# them into a DataFrame.

def load_json_to_df(filepath):
    data = []
    with open(filepath) as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    return df