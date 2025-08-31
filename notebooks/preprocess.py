# --- Step 1: Import Libraries ---
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MultiLabelBinarizer

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(r"C:\AI-Symptom-Checker\data\processed", exist_ok=True)
    print("Created data/processed directory")

def load_datasets():
    """Load and validate datasets."""
    try:
        # Use raw strings to handle Windows paths correctly
        disease_symptoms = pd.read_csv(r"C:\AI-Symptom-Checker\data\DiseaseAndSymptoms.csv")
        disease_precautions = pd.read_csv(r"C:\AI-Symptom-Checker\data\Disease precaution.csv")
        
        print("✓ Datasets loaded successfully")
        print(f"Disease-Symptoms shape: {disease_symptoms.shape}")
        print(f"Disease-Precautions shape: {disease_precautions.shape}")
        
        return disease_symptoms, disease_precautions
        
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure the CSV files are in the 'data/' directory")
        return None, None

def clean_datasets(disease_symptoms, disease_precautions):
    """Basic cleaning operations."""
    print("\n--- Cleaning Datasets ---")
    
    # Check for duplicates before removing
    print(f"Disease-Symptoms duplicates: {disease_symptoms.duplicated().sum()}")
    print(f"Disease-Precautions duplicates: {disease_precautions.duplicated().sum()}")
    
    # Drop duplicates
    disease_symptoms = disease_symptoms.drop_duplicates()
    disease_precautions = disease_precautions.drop_duplicates()
    
    # Check for missing values
    print(f"\nMissing values in Disease-Symptoms:\n{disease_symptoms.isnull().sum()}")
    print(f"\nMissing values in Disease-Precautions:\n{disease_precautions.isnull().sum()}")
    
    # Handle missing values
    disease_symptoms = disease_symptoms.fillna("")
    disease_precautions = disease_precautions.fillna("")
    
    print("✓ Datasets cleaned")
    return disease_symptoms, disease_precautions

def process_symptoms_dataset(disease_symptoms):
    """Process symptoms dataset with multi-hot encoding."""
    print("\n--- Processing Symptoms Dataset ---")
    
    # Get symptom columns (all columns except 'Disease')
    symptom_cols = [col for col in disease_symptoms.columns if col != 'Disease']
    print(f"Found {len(symptom_cols)} symptom columns: {symptom_cols[:5]}...")
    
    # Convert symptom columns to list format for each row
    def extract_symptoms(row):
        symptoms = []
        for col in symptom_cols:
            if pd.notna(row[col]) and str(row[col]).strip():
                symptoms.append(str(row[col]).strip())
        return symptoms
    
    # Apply symptom extraction
    disease_symptoms['Symptoms'] = disease_symptoms.apply(extract_symptoms, axis=1)
    
    # Check for empty symptom lists
    empty_symptoms = disease_symptoms['Symptoms'].apply(len) == 0
    if empty_symptoms.any():
        print(f"Warning: {empty_symptoms.sum()} diseases have no symptoms listed")
    
    # Get all unique symptoms
    all_symptoms = set()
    for symptom_list in disease_symptoms['Symptoms']:
        all_symptoms.update(symptom_list)
    
    print(f"Found {len(all_symptoms)} unique symptoms")
    
    # Multi-hot encode symptoms
    mlb = MultiLabelBinarizer()
    symptom_features = mlb.fit_transform(disease_symptoms['Symptoms'])
    symptom_df = pd.DataFrame(symptom_features, columns=mlb.classes_)
    
    # Remove empty string column if it exists
    if '' in symptom_df.columns:
        symptom_df = symptom_df.drop('', axis=1)
    
    # Combine with disease column
    processed_symptoms = pd.concat([disease_symptoms['Disease'], symptom_df], axis=1)
    
    print(f"✓ Processed {len(mlb.classes_)} unique symptoms")
    print(f"✓ Final shape: {processed_symptoms.shape}")
    
    return processed_symptoms, mlb

def process_precautions_dataset(disease_precautions):
    """Process precautions dataset."""
    print("\n--- Processing Precautions Dataset ---")
    
    # Get precaution columns (exclude 'Disease' column)
    precaution_cols = [col for col in disease_precautions.columns if col != 'Disease']
    
    # Process precautions for each disease
    processed_data = []
    for _, row in disease_precautions.iterrows():
        disease = row['Disease']
        precautions = [str(row[col]).strip() for col in precaution_cols 
                      if pd.notna(row[col]) and str(row[col]).strip() != '']
        processed_data.append({'Disease': disease, 'Precautions': precautions})
    
    processed_precautions = pd.DataFrame(processed_data)
    
    # Check for diseases with no precautions
    empty_precautions = processed_precautions['Precautions'].apply(len) == 0
    if empty_precautions.any():
        print(f"Warning: {empty_precautions.sum()} diseases have no precautions listed")
    
    print(f"✓ Processed precautions for {len(processed_precautions)} diseases")
    
    return processed_precautions

def validate_data_consistency(processed_symptoms, processed_precautions):
    """Check consistency between datasets."""
    print("\n--- Validating Data Consistency ---")
    
    symptoms_diseases = set(processed_symptoms['Disease'].unique())
    precautions_diseases = set(processed_precautions['Disease'].unique())
    
    common_diseases = symptoms_diseases.intersection(precautions_diseases)
    symptoms_only = symptoms_diseases - precautions_diseases
    precautions_only = precautions_diseases - symptoms_diseases
    
    print(f"Common diseases: {len(common_diseases)}")
    print(f"Diseases only in symptoms: {len(symptoms_only)}")
    print(f"Diseases only in precautions: {len(precautions_only)}")
    
    if symptoms_only:
        print(f"Diseases without precautions: {list(symptoms_only)[:5]}...")
    if precautions_only:
        print(f"Diseases without symptoms: {list(precautions_only)[:5]}...")

def save_processed_data(processed_symptoms, processed_precautions, mlb):
    """Save processed datasets and metadata."""
    print("\n--- Saving Processed Data ---")
    
    # Save processed datasets
    processed_symptoms.to_csv(r"C:\AI-Symptom-Checker\data\processed\disease_symptoms_processed.csv", index=False)
    processed_precautions.to_csv(r"C:\AI-Symptom-Checker\data\processed\disease_precautions_processed.csv", index=False)
    
    # Save symptom vocabulary for future use
    symptom_vocab = pd.DataFrame({'symptoms': mlb.classes_})
    symptom_vocab.to_csv(r"C:\AI-Symptom-Checker\data\processed\symptom_vocabulary.csv", index=False)
    
    print("✓ Processed datasets saved:")
    print("  - data/processed/disease_symptoms_processed.csv")
    print("  - data/processed/disease_precautions_processed.csv")
    print("  - data/processed/symptom_vocabulary.csv")

def main():
    """Main preprocessing pipeline."""
    print("=== AI Symptom Checker Data Preprocessing ===\n")
    
    # Create directories
    create_directories()
    
    # Load datasets
    disease_symptoms, disease_precautions = load_datasets()
    if disease_symptoms is None or disease_precautions is None:
        return
    
    # Display sample data
    print("\nDisease-Symptoms sample:")
    print(disease_symptoms.head())
    print("\nDisease-Precautions sample:")
    print(disease_precautions.head())
    
    # Clean datasets
    disease_symptoms, disease_precautions = clean_datasets(disease_symptoms, disease_precautions)
    
    # Process datasets
    processed_symptoms, mlb = process_symptoms_dataset(disease_symptoms)
    processed_precautions = process_precautions_dataset(disease_precautions)
    
    # Validate consistency
    validate_data_consistency(processed_symptoms, processed_precautions)
    
    # Save processed data
    save_processed_data(processed_symptoms, processed_precautions, mlb)
    
    print("\n=== Preprocessing Complete! ===")

if __name__ == "__main__":
    main()