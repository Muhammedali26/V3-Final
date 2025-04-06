import os
import pandas as pd
import json
from datetime import datetime

# Load CSV data
print("Loading synthetic data...")
try:
    df = pd.read_csv('synthetic_data_indexed.csv')
    print(f"Loaded CSV with {len(df)} rows and columns: {list(df.columns)}")
    
    # Check for expected columns
    required_columns = ['Goal', 'Issue', 'Hair Type', 'Hair Texture', 'Hair Behaviour', 
                         'Scalp Feeling', 'DRYNESS', 'DAMAGE', 'SENSITIVITY', 
                         'SEBUM Oil', 'DRY SCALP', 'FLAKES']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing columns in CSV: {missing_columns}")
    
    # Rename columns to match expected format
    column_mapping = {
        'Hair Type': 'Hair_Type',
        'Hair Texture': 'Hair_Texture',
        'Hair Behaviour': 'Hair_Behaviour',
        'Scalp Feeling': 'Scalp_Feeling',
        'SEBUM Oil': 'SEBUM_Oil',
        'DRY SCALP': 'DRY_SCALP',
        'Base_Name': 'base_name',
        'Unique_ID': 'unique_id'
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    print(f"Renamed columns. New columns: {list(df.columns)}")
    
    # Convert to list of dictionaries for JSON
    preferences = df.to_dict(orient='records')
    print(f"Converted to {len(preferences)} preference records")
    
    # Save to preferences.json
    with open('preferences.json', 'w') as f:
        json.dump(preferences, f, indent=4)
    print(f"Saved {len(preferences)} records to preferences.json")
    
    # Make HTTP request to batch_update endpoint
    import requests
    print("Calling batch_update endpoint...")
    try:
        response = requests.post('http://localhost:8002/batch_update', json={'min_samples': 1})
        print(f"Batch update response: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Update result: {result}")
        else:
            print(f"Error: {response.text}")
            print("\nSince the server might not be running, let's trigger a manual model update by creating a script:")
            print("To complete the update, run the app and call the /batch_update endpoint")
    except Exception as e:
        print(f"Error calling batch_update: {e}")
        print("The server is likely not running. Start the app and call the /batch_update endpoint")
        
except Exception as e:
    print(f"Error: {e}") 