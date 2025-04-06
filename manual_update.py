import os
import json
import pickle
import numpy as np
import scipy.sparse
import pandas as pd
import re
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

print("Starting manual model update process...")

# Load preferences
try:
    with open('preferences.json', 'r') as f:
        preferences_data = json.load(f)
    print(f"Loaded {len(preferences_data)} preferences from preferences.json")
    
    # Convert to DataFrame
    preferences_df = pd.DataFrame(preferences_data)
    print(f"Converted to DataFrame with columns: {preferences_df.columns.tolist()}")
    
    # Define feature columns
    feature_columns = ['DRYNESS', 'DAMAGE', 'SENSITIVITY', 'SEBUM_Oil', 'DRY_SCALP', 'FLAKES']
    print(f"Using feature columns: {feature_columns}")
    
    # Ensure all required columns exist
    for col in feature_columns:
        if col not in preferences_df.columns:
            preferences_df[col] = 0.0
            print(f"Added missing column {col} with default values")
            
    # Extract numeric features
    X_numeric = preferences_df[feature_columns].fillna(0.0).values
    print(f"Numeric features shape: {X_numeric.shape}")
    
    # Create and fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    print(f"Scaled numeric features shape: {X_scaled.shape}")
    
    # Extract text features
    text_columns = ['Goal', 'Issue', 'Hair_Type', 'Hair_Texture', 
                  'Hair_Behaviour', 'Scalp_Feeling']
    
    # Create combined text features
    def combine_text_with_user_info(row):
        text_features = []
        for col in text_columns:
            if col in row and pd.notna(row[col]):
                val = str(row[col]).strip()
                if val and val.lower() != 'unknown' and val.lower() != 'nan':
                    text_features.append(val)
        
        # Add user identification if available
        if 'base_name' in row and pd.notna(row['base_name']):
            text_features.append(str(row['base_name']))
        if 'unique_id' in row and pd.notna(row['unique_id']):
            text_features.append(str(row['unique_id']))
        
        # Join everything and clean
        text = ' '.join(text_features)
        # Remove special characters and extra spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    # Apply function to create combined text
    text_data = preferences_df.apply(combine_text_with_user_info, axis=1)
    non_empty_texts = [t for t in text_data if t.strip()]
    
    if not non_empty_texts:
        print("Error: No valid text data found in preferences")
        exit(1)
    
    print(f"Created {len(non_empty_texts)} text features")
    print(f"Text sample: {non_empty_texts[0][:100]}...")
    
    # Create vectorizer
    vectorizer = TfidfVectorizer(
        stop_words=None,
        ngram_range=(1, 2),
        max_features=2000,
        min_df=1,
        use_idf=True,
        sublinear_tf=True,
        analyzer='word',
        token_pattern=r'(?u)\b\w+\b'
    )
    
    # Transform text data
    X_text = vectorizer.fit_transform(non_empty_texts)
    print(f"Vectorized text features shape: {X_text.shape}")
    
    # Combine features
    if scipy.sparse.issparse(X_text):
        from scipy.sparse import hstack
        X_combined = hstack([X_text, scipy.sparse.csr_matrix(X_scaled)])
        print(f"Using sparse hstack - X_combined shape: {X_combined.shape}")
    else:
        X_combined = np.hstack([X_text.toarray(), X_scaled])
        print(f"Using numpy hstack - X_combined shape: {X_combined.shape}")
    
    # Adjust neighbors
    n_neighbors = min(10, len(preferences_data))
    
    # Create KNN model
    knn_model = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm='brute',
        n_jobs=-1
    )
    
    # Fit model
    knn_model.fit(X_combined)
    print(f"Fitted NearestNeighbors model with {knn_model.n_samples_fit_} samples and {n_neighbors} neighbors")
    
    # Save models
    with open('nn_model.pkl', 'wb') as f:
        pickle.dump(knn_model, f)
    print("Saved KNN model to nn_model.pkl")
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Saved scaler to scaler.pkl")
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("Saved vectorizer to vectorizer.pkl")
    
    # Save feature info
    feature_info = {
        "feature_columns": feature_columns,
        "feature_count": X_numeric.shape[1],
        "timestamp": datetime.now().isoformat()
    }
    
    with open('feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=4)
    print("Saved feature info to feature_info.json")
    
    print("\nModel update completed successfully!")
    
except Exception as e:
    import traceback
    print(f"Error updating model: {str(e)}")
    print(traceback.format_exc()) 