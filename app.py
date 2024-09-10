import streamlit as st
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import spacy

# Initialize the models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

# Function to generate embeddings
def generate_embeddings_batch(texts):
    return sentence_model.encode(texts, convert_to_numpy=True)

# Function to preprocess query
def preprocess_query(query):
    query = query.lower().strip()
    return query

# Function to extract factors and context using NER
def extract_factors_and_context(query):
    factors = {
        'geographic': None,
        'therapeutic': None,
        'dataset': None
    }

    doc = nlp(query)
    
    # Extract named entities
    for ent in doc.ents:
        if ent.label_ in ['GPE', 'LOC']:  # GPE: Geopolitical Entity, LOC: Location
            factors['geographic'] = ent.text
        elif ent.label_ in ['ORG', 'PRODUCT', 'EVENT', 'WORK_OF_ART']:
            factors['therapeutic'] = ent.text
    
    # The remaining query is considered as dataset-related information
    if query:
        factors['dataset'] = query

    return factors

# Function to search datasets based on factors and context clues
def search_datasets(query, df_cleaned, factors):
    query_embedding = generate_embeddings_batch([query])[0]
    
    relevant_columns = []
    if factors['geographic']:
        relevant_columns.append('Geographic coverage')
    if factors['therapeutic']:
        relevant_columns.append('Therapeutic coverage')
    if factors['dataset']:
        relevant_columns.append('Dataset Category')
    
    # Combine the relevant columns into a single string
    combined_coverage = df_cleaned[relevant_columns].fillna('').agg(' '.join, axis=1)
    column_embeddings = generate_embeddings_batch(combined_coverage.tolist())
    
    # Calculate similarities
    similarities = np.inner(query_embedding, column_embeddings)
    df_cleaned['similarity'] = similarities

    # Apply penalty for "TBD" values in specific columns
    penalty_columns = [
        'Dataset Category',
        'Vendor',
        'Dataset Name',
        'Description',
        'Therapeutic coverage',
        'Geographic coverage'
    ]
    
    def apply_penalty(row):
        penalty = 0
        for column in penalty_columns:
            if 'TBD' in str(row.get(column, '')):
                penalty += 0.1  # Adjust penalty as needed
        return penalty
    
    df_cleaned['penalty'] = df_cleaned.apply(apply_penalty, axis=1)
    df_cleaned['penalized_similarity'] = df_cleaned['similarity'] - df_cleaned['penalty']
    
    # Create masks for filtering
    geo_mask = df_cleaned['Geographic coverage'].str.contains(factors['geographic'], case=False, na=False) if factors['geographic'] else np.ones(len(df_cleaned), dtype=bool)
    category_mask = df_cleaned['Dataset Category'].str.contains(factors['dataset'], case=False, na=False) if factors['dataset'] else np.ones(len(df_cleaned), dtype=bool)
    
    # Apply masks
    filtered_by_geo_and_cat = df_cleaned[geo_mask & category_mask]
    
    # Determine top results
    top_k = 20
    if not filtered_by_geo_and_cat.empty:
        top_results = filtered_by_geo_and_cat.nlargest(top_k, 'penalized_similarity')
    else:
        top_results = df_cleaned.nlargest(top_k, 'penalized_similarity')
    
    return top_results

# Load and preprocess data
@st.cache_data
def load_data():
    file_path = 'my_dap.csv'
    df_cleaned = pd.read_csv(file_path)

    columns = ['Dataset Category', 'Vendor', 'Dataset Name', 'Description', 
               'Therapeutic coverage', 'Indication coverage', 'Geographic coverage']

    embeddings = {}
    for col in columns:
        texts = df_cleaned[col].astype(str).tolist()
        col_embeddings = generate_embeddings_batch(texts)
        embeddings[col] = col_embeddings

    for col in embeddings:
        df_cleaned[f'{col}_embedding'] = list(map(lambda x: ' '.join(map(str, x)), embeddings[col]))

    df_cleaned.to_csv('processed_data_with_embeddings.csv', index=False)

    return df_cleaned

df_cleaned = load_data()

# Streamlit UI
st.title("Dataset Search Application")

query = st.text_input("Enter your query")

if st.button("Search"):
    if query:
        processed_query = preprocess_query(query)
        factors = extract_factors_and_context(processed_query)
        
        st.write(f"Extracted Factors and Context: {factors}")
        
        top_results = search_datasets(processed_query, df_cleaned, factors)
        
        if top_results.empty:
            st.write("No results found.")
        else:
            st.write("Top Results:")
            for _, row in top_results.iterrows():
                st.write(f"Dataset Category: {row['Dataset Category']}")
                st.write(f"Vendor: {row['Vendor']}")
                st.write(f"Dataset Name: {row['Dataset Name']}")
                st.write(f"Description: {row['Description']}")
                st.write(f"Therapeutic coverage: {row['Therapeutic coverage']}")
                st.write(f"Geographic coverage: {row['Geographic coverage']}")
                st.write("---")
    else:
        st.write("Please enter a query.")
