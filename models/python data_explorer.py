import pandas as pd
import os

def explore_dataset(file_path):
    """Explore a single dataset"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        print(f"\n{'='*60}")
        print(f"FILE: {file_path}")
        print(f"{'='*60}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        print(f"\nData types:")
        print(df.dtypes)
        print(f"\nMissing values:")
        print(df.isnull().sum())
        
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Explore all your datasets
data_files = [
    r'C:\AI-Symptom-Checker\data\processed\disease_precautions_processed.csv',
    r'C:\AI-Symptom-Checker\data\processed\disease_symptoms_processed.csv', 
    r'C:\AI-Symptom-Checker\data\processed\symptom_vocabulary.csv'
]

datasets = {}
for file_path in data_files:
    df = explore_dataset(file_path)
    if df is not None:
        datasets[os.path.basename(file_path)] = df

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Successfully loaded {len(datasets)} datasets")
for name, df in datasets.items():
    print(f"{name}: {df.shape[0]} rows, {df.shape[1]} columns")