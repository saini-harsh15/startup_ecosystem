# investor_embeddings.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

def create_investor_embeddings():
    """
    Reads investor CSV, generates embeddings using Sentence-BERT,
    and saves FAISS index and investor DataFrame for fast retrieval.
    """
    
    print("Loading investor dataset...")
    
    # Load the investor dataset
    try:
        df = pd.read_csv('investor_dataset_final.csv')
        print(f"Loaded {len(df)} investors from dataset")
    except FileNotFoundError:
        print(" Error: investor_dataset_final.csv not found!")
        print("Please ensure the CSV file is in the same directory.")
        return
    
    # Initialize Sentence-BERT model
    print(" Loading Sentence-BERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(" Model loaded successfully")
    
    # Prepare text data for each investor
    print("Preparing investor profiles for embedding...")
    
    def create_investor_profile(row):
        """
        Combines investor information into a single text profile
        for semantic embedding.
        """
        profile_parts = []
        
        # Add domains (main focus areas)
        if pd.notna(row['preferred_domains']):
            profile_parts.append(f"Investment domains: {row['preferred_domains']}")
        
        # Add funding stages
        if pd.notna(row['funding_stages']):
            profile_parts.append(f"Funding stages: {row['funding_stages']}")
        
        # Add investment range
        if pd.notna(row['investment_range_usd']):
            profile_parts.append(f"Investment range: {row['investment_range_usd']}")
        
        # Add location
        if pd.notna(row['location']):
            profile_parts.append(f"Location: {row['location']}")
        
        # Add description (investment thesis)
        if pd.notna(row['description']):
            profile_parts.append(f"Investment thesis: {row['description']}")
        
        return " | ".join(profile_parts)
    
    # Create profile texts
    investor_profiles = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        profile = create_investor_profile(row)
        if profile.strip():  # Only include non-empty profiles
            investor_profiles.append(profile)
            valid_indices.append(idx)
    
    print(f"Created {len(investor_profiles)} investor profiles")
    
    # Filter dataframe to only valid investors
    df_filtered = df.iloc[valid_indices].reset_index(drop=True)
    
    # Generate embeddings
    print(" Generating embeddings (this may take a few minutes)...")
    embeddings = model.encode(investor_profiles, show_progress_bar=True)
    print(f" Generated embeddings with shape: {embeddings.shape}")
    
    # Create FAISS index
    print(" Creating FAISS index...")
    dimension = embeddings.shape[1]  # Should be 384 for all-MiniLM-L6-v2
    
    # Use IndexFlatIP for cosine similarity (Inner Product)
    # We'll normalize vectors so IP becomes cosine similarity
    index = faiss.IndexFlatIP(dimension)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings.astype(np.float32))
    
    # Add embeddings to index
    index.add(embeddings.astype(np.float32))
    print(f"FAISS index created with {index.ntotal} vectors")
    
    # Save everything
    print("Saving index and data...")
    
    # Save FAISS index
    faiss.write_index(index, 'investor_index.faiss')
    
    # Save investor dataframe and profiles
    with open('investor_data.pkl', 'wb') as f:
        pickle.dump({
            'dataframe': df_filtered,
            'profiles': investor_profiles
        }, f)
    
    print(" Investor embeddings and index created successfully!")
    print("Files created:")
    print("  - investor_index.faiss (FAISS similarity index)")
    print("  - investor_data.pkl (investor dataframe and profiles)")
    
    return True

if __name__ == "__main__":
    create_investor_embeddings()
