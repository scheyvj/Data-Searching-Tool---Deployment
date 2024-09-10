import pandas as pd
from model import generate_embedding  # Import the function to generate embeddings


import pandas as pd

file_path = 'genai_data.csv'

df = pd.read_csv(file_path)
new_df = df.iloc[:, [0, 1, 2, 3, 5, 8, 28]]
nnew_df = new_df.dropna(subset=['Description'])
nnew_df = nnew_df.dropna(subset=['TA coverage'])
df_cleaned = nnew_df

def concatenate_columns(row):
    return ' '.join([
        str(row['Country']),
        str(row['Region']),
        str(row['Name of Database/Report']),
        str(row['Dataset Type']),
        str(row['Parent Vendor Name (If Applicable)']),
        str(row['Description']),
        str(row['TA coverage'])
    ])

# Add concatenated column
df_cleaned['concatenated'] = df_cleaned.apply(concatenate_columns, axis=1)

# Generate embeddings
df_cleaned['embedding'] = df_cleaned['concatenated'].apply(lambda x: generate_embedding(x))

# Save the dataframe with embeddings
df_cleaned.to_csv('processed_data_with_embeddings.csv', index=False)
