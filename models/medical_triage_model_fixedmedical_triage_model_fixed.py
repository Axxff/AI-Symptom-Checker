import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os

class ImprovedMedicalTriageModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.symptom_columns = None
        self.feature_importance = None
        
    def load_data(self):
        """Load and preprocess the dataset"""
        file_path = r'C:\AI-Symptom-Checker\data\processed\disease_symptoms_processed.csv'
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found: {file_path}")
        
        df = pd.read_csv(file_path)
        print(f"Original dataset shape: {df.shape}")
        
        # Remove rows where Disease is NaN
        df_clean = df.dropna(subset=['Disease']).copy()
        print(f"After removing NaN diseases: {df_clean.shape}")
        
        # Fill NaN symptom values with 0
        symptom_cols = [col for col in df_clean.columns if col != 'Disease']
        df_clean[symptom_cols] = df_clean[symptom_cols].fillna(0)
        
        # Remove diseases with very few samples (less than 3)
        disease_counts = df_clean['Disease'].value_counts()
        diseases_to_keep = disease_counts[disease_counts >= 3].index
        df_filtered = df_clean[df_clean['Disease'].isin(diseases_to_keep)].copy()
        
        print(f"After filtering rare diseases: {df_filtered.shape}")
        print(f"Remaining diseases: {df_filtered['Disease'].nunique()}")
        
        return df_filtered
    
    def feature_engineering(self, df):
        """Create additional features"""
        symptom_cols = [col for col in df.columns if col != 'Disease']
        
        # Add feature: total number of symptoms
        df['total_symptoms'] = df[symptom_cols].sum(axis=1)
        
        # Add symptom category features
        pain_symptoms = [col for col in symptom_cols if 'pain' in col]
        fever_symptoms = [col for col in symptom_cols if 'fever' in col]
        digestive_symptoms = [col for col in symptom_cols if any(word in col for word in ['stomach', 'nausea', 'vomit', 'diarrhoea', 'constipation'])]
        respiratory_symptoms = [col for col in symptom_cols if any(word in col for word in ['cough', 'breathing', 'chest', 'congestion'])]
        
        df['pain_score'] = df[pain_symptoms].sum(axis=1)
        df['fever_score'] = df[fever_symptoms].sum(axis=1)  
        df['digestive_score'] = df[digestive_symptoms].sum(axis=1)
        df['respiratory_score'] = df[respiratory_symptoms].sum(axis=1)
        
        return df
    
    def prepare_data(self, df):
        """Prepare data for training"""
        # Apply feature engineering
        df_engineered = self.feature_engineering(df)
        
        # Separate features and target
        feature_cols = [col for col in df_engineered.columns if col != 'Disease']
        self.symptom_columns = feature_cols
        
        X = df_engineered[feature_cols].values
        y = df_engineered['Disease'].values
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Feature matrix shape: {X_scaled.shape}")
        print(f"Target vector shape: {y.shape}")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_multiple_models(self, X_train, y_train):
        """Train and compare multiple models"""
        # Calculate class weights to handle imbalanced data
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=3,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=42,
                multi_class='ovr'
            )
        }
        
        best_score = 0
        best_model = None
        best_name = ""
        
        print("Training multiple models...")
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            # Cross-validation score
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            avg_cv_score = cv_scores.mean()
            
            print(f"{name} CV Score: {avg_cv_score:.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            if avg_cv_score > best_score:
                best_score = avg_cv_score
                best_model = model
                best_name = name
        
        self.model = best_model
        print(f"\nBest model: {best_name} with CV score: {best_score:.3f}")
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
        
        return best_name
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest Accuracy: {accuracy:.3f}")
        
        # Show detailed metrics for top diseases
        from sklearn.metrics import classification_report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        return accuracy
    
    def show_feature_importance(self, top_n=20):
        """Show most important features"""
        if self.feature_importance is not None:
            feature_importance_df = pd.DataFrame({
                'feature': self.symptom_columns,
                'importance': self.feature_importance
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop {top_n} Most Important Features:")
            for i, row in feature_importance_df.head(top_n).iterrows():
                feature_name = row['feature'].replace('_', ' ').title()
                print(f"{feature_name}: {row['importance']:.4f}")
    
    def predict_from_text(self, symptom_text):
        """Enhanced text-based prediction"""
        symptom_text = symptom_text.lower()
        
        # Create feature vector (excluding engineered features for now)
        base_symptom_cols = [col for col in self.symptom_columns if not col in ['total_symptoms', 'pain_score', 'fever_score', 'digestive_score', 'respiratory_score']]
        feature_vector = np.zeros(len(base_symptom_cols))
        
        for i, symptom_col in enumerate(base_symptom_cols):
            readable_symptom = symptom_col.replace('_', ' ')
            if readable_symptom in symptom_text or symptom_col in symptom_text:
                feature_vector[i] = 1
        
        # Calculate engineered features
        total_symptoms = np.sum(feature_vector)
        pain_symptoms = sum(1 for col in base_symptom_cols if 'pain' in col and feature_vector[base_symptom_cols.index(col)] == 1)
        fever_symptoms = sum(1 for col in base_symptom_cols if 'fever' in col and feature_vector[base_symptom_cols.index(col)] == 1)
        digestive_symptoms = sum(1 for col in base_symptom_cols if any(word in col for word in ['stomach', 'nausea', 'vomit', 'diarrhoea', 'constipation']) and feature_vector[base_symptom_cols.index(col)] == 1)
        respiratory_symptoms = sum(1 for col in base_symptom_cols if any(word in col for word in ['cough', 'breathing', 'chest', 'congestion']) and feature_vector[base_symptom_cols.index(col)] == 1)
        
        # Combine all features
        full_feature_vector = np.concatenate([
            feature_vector,
            [total_symptoms, pain_symptoms, fever_symptoms, digestive_symptoms, respiratory_symptoms]
        ])
        
        # Scale features
        full_feature_vector = self.scaler.transform(full_feature_vector.reshape(1, -1))
        
        # Get prediction
        prediction = self.model.predict(full_feature_vector)[0]
        probabilities = self.model.predict_proba(full_feature_vector)[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_conditions = [self.model.classes_[i] for i in top_indices]
        top_probabilities = [probabilities[i] for i in top_indices]
        
        return {
            'top_prediction': prediction,
            'top_conditions': top_conditions,
            'probabilities': top_probabilities,
            'symptoms_detected': int(total_symptoms)
        }
    
    def save_model(self, filename='improved_medical_triage_model'):
        """Save the trained model and all components"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'symptom_columns': self.symptom_columns,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, f'{filename}.pkl')
        print(f"Improved model saved as {filename}.pkl")

# Training script
if __name__ == "__main__":
    try:
        # Initialize improved model
        triage_model = ImprovedMedicalTriageModel()
        
        # Load and preprocess data
        df = triage_model.load_data()
        
        # Prepare data
        X_train, X_test, y_train, y_test = triage_model.prepare_data(df)
        
        # Train multiple models and select best
        best_model_name = triage_model.train_multiple_models(X_train, y_train)
        
        # Evaluate model
        accuracy = triage_model.evaluate_model(X_test, y_test)
        
        # Show feature importance
        triage_model.show_feature_importance()
        
        # Save model
        triage_model.save_model()
        
        print("\n" + "="*60)
        print("TESTING IMPROVED MODEL")
        print("="*60)
        
        # Test cases
        test_cases = [
            "headache and high fever",
            "chest pain and breathlessness", 
            "stomach pain and vomiting",
            "skin rash and itching",
            "joint pain and fatigue"
        ]
        
        for test_case in test_cases:
            result = triage_model.predict_from_text(test_case)
            print(f"\nTest: '{test_case}'")
            print(f"Prediction: {result['top_prediction']} ({result['probabilities'][0]:.1%})")
            print(f"Top 3: {', '.join([f'{c} ({p:.1%})' for c, p in zip(result['top_conditions'], result['probabilities'])])}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()