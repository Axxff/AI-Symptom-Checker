import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import pickle
import joblib

class MedicalTriageModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.model = None
        self.label_encoder = None
        
    def prepare_data(self, df, symptom_col='symptoms_normalized', condition_col='condition_clean'):
        """Prepare data for training"""
        # Extract symptoms and conditions
        X = df[symptom_col].values
        y = df[condition_col].values
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, model_type='random_forest'):
        """Train the model"""
        print("Vectorizing symptoms...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        
        print(f"Training {model_type} model...")
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'logistic_regression':
            self.model = OneVsRestClassifier(
                LogisticRegression(random_state=42, max_iter=1000)
            )
        
        self.model.fit(X_train_tfidf, y_train)
        print("Model training completed!")
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        X_test_tfidf = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_tfidf)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.3f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def predict(self, symptom_text):
        """Make prediction for new symptom text"""
        if isinstance(symptom_text, str):
            symptom_text = [symptom_text]
        
        # Vectorize input
        symptom_tfidf = self.vectorizer.transform(symptom_text)
        
        # Get prediction and probability
        prediction = self.model.predict(symptom_tfidf)[0]
        probabilities = self.model.predict_proba(symptom_tfidf)[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_conditions = [self.model.classes_[i] for i in top_indices]
        top_probabilities = [probabilities[i] for i in top_indices]
        
        return {
            'top_prediction': prediction,
            'top_conditions': top_conditions,
            'probabilities': top_probabilities
        }
    
    def save_model(self, filename='medical_triage_model'):
        """Save trained model and vectorizer"""
        joblib.dump(self.model, f'{filename}_model.pkl')
        joblib.dump(self.vectorizer, f'{filename}_vectorizer.pkl')
        print(f"Model saved as {filename}_model.pkl")
        print(f"Vectorizer saved as {filename}_vectorizer.pkl")
    
    def load_model(self, filename='medical_triage_model'):
        """Load pre-trained model and vectorizer"""
        self.model = joblib.load(f'{filename}_model.pkl')
        self.vectorizer = joblib.load(f'{filename}_vectorizer.pkl')
        print("Model loaded successfully!")

# Training script
if __name__ == "__main__":
    # Load processed data
    df = pd.read_csv('C:\AI-Symptom-Checker\data\processed\disease_precautions_processed.csv')
    df = pd.read_csv('C:\AI-Symptom-Checker\data\processed\disease_symptoms_processed.csv')
    df = pd.read_csv('C:\AI-Symptom-Checker\data\processed\symptom_vocabulary.csv')

    # Initialize model
    triage_model = MedicalTriageModel()
    
    # Prepare data
    X_train, X_test, y_train, y_test = triage_model.prepare_data(df)
    
    # Train model
    triage_model.train_model(X_train, y_train, model_type='random_forest')
    
    # Evaluate model
    accuracy = triage_model.evaluate_model(X_test, y_test)
    
    # Save model
    triage_model.save_model()
    
    # Test prediction
    test_symptoms = "headache and fever"
    result = triage_model.predict(test_symptoms)
    print(f"\nTest prediction for '{test_symptoms}':")
    print(f"Top condition: {result['top_prediction']}")
    print(f"Top 3 conditions: {result['top_conditions']}")
    print(f"Probabilities: {result['probabilities']}")