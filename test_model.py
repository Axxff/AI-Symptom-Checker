"""
Simple script to test the trained model - Fixed version
"""
import pandas as pd
import numpy as np
import joblib

def load_and_test_model():
    """Load the trained model and test it"""
    try:
        # Load the saved model
        model_data = joblib.load('medical_triage_model.pkl')
        model = model_data['model']
        symptom_columns = model_data['symptom_columns']
        
        print("Model loaded successfully!")
        print(f"Number of symptoms: {len(symptom_columns)}")
        print(f"Number of diseases the model can predict: {len(model.classes_)}")
        print("="*50)
        
        def predict_from_text(symptom_text):
            """Make prediction from text description"""
            symptom_text = symptom_text.lower()
            
            # Create feature vector
            feature_vector = np.zeros(len(symptom_columns))
            
            for i, symptom_col in enumerate(symptom_columns):
                readable_symptom = symptom_col.replace('_', ' ')
                if readable_symptom in symptom_text or symptom_col in symptom_text:
                    feature_vector[i] = 1
            
            feature_vector = feature_vector.reshape(1, -1)
            
            # Get prediction and probabilities
            prediction = model.predict(feature_vector)[0]
            probabilities = model.predict_proba(feature_vector)[0]
            
            # Get top 3 predictions
            top_indices = np.argsort(probabilities)[-3:][::-1]
            top_conditions = [model.classes_[i] for i in top_indices]
            top_probabilities = [probabilities[i] for i in top_indices]
            
            return {
                'top_prediction': prediction,
                'top_conditions': top_conditions,
                'probabilities': top_probabilities,
                'symptoms_detected': int(np.sum(feature_vector))
            }
        
        def predict_from_symptoms_dict(symptoms_dict):
            """Make prediction from symptom dictionary"""
            feature_vector = np.zeros(len(symptom_columns))
            
            for symptom, present in symptoms_dict.items():
                symptom_formatted = symptom.lower().replace(' ', '_').replace('-', '_')
                
                if symptom_formatted in symptom_columns:
                    idx = symptom_columns.index(symptom_formatted)
                    feature_vector[idx] = present
                else:
                    # Try alternative formats
                    for i, col in enumerate(symptom_columns):
                        if symptom_formatted in col or col in symptom_formatted:
                            feature_vector[i] = present
                            break
            
            feature_vector = feature_vector.reshape(1, -1)
            
            # Get prediction and probabilities
            prediction = model.predict(feature_vector)[0]
            probabilities = model.predict_proba(feature_vector)[0]
            
            # Get top 3 predictions
            top_indices = np.argsort(probabilities)[-3:][::-1]
            top_conditions = [model.classes_[i] for i in top_indices]
            top_probabilities = [probabilities[i] for i in top_indices]
            
            return {
                'top_prediction': prediction,
                'top_conditions': top_conditions,
                'probabilities': top_probabilities,
                'symptoms_detected': int(np.sum(feature_vector))
            }
        
        # Interactive testing
        while True:
            print("\nChoose testing method:")
            print("1. Enter symptoms as text (e.g., 'headache and fever')")
            print("2. Try predefined test cases")
            print("3. Show available symptoms")
            print("4. Exit")
            
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                symptom_text = input("Enter your symptoms: ").strip()
                if symptom_text:
                    result = predict_from_text(symptom_text)
                    print(f"\nPrediction Results for: '{symptom_text}'")
                    print(f"Symptoms detected: {result['symptoms_detected']}")
                    print(f"Top condition: {result['top_prediction']}")
                    print(f"Confidence: {result['probabilities'][0]:.1%}")
                    print(f"\nTop 3 possibilities:")
                    for i, (condition, prob) in enumerate(zip(result['top_conditions'], result['probabilities'])):
                        print(f"{i+1}. {condition}: {prob:.1%}")
            
            elif choice == '2':
                test_cases = [
                    "fever and headache",
                    "cough and chest pain",
                    "stomach pain and nausea",
                    "high fever and chills",
                    "skin rash and itching",
                    "joint pain and fatigue"
                ]
                
                for test_case in test_cases:
                    result = predict_from_text(test_case)
                    print(f"\nTest: '{test_case}'")
                    print(f"Prediction: {result['top_prediction']} ({result['probabilities'][0]:.1%})")
                    print(f"Symptoms detected: {result['symptoms_detected']}")
            
            elif choice == '3':
                print(f"\nAvailable symptoms ({len(symptom_columns)} total):")
                for i, symptom in enumerate(symptom_columns[:30]):  # Show first 30
                    readable = symptom.replace('_', ' ').title()
                    print(f"{i+1:2}. {readable}")
                if len(symptom_columns) > 30:
                    print(f"... and {len(symptom_columns) - 30} more")
                
                print(f"\nAvailable diseases ({len(model.classes_)} total):")
                for i, disease in enumerate(sorted(model.classes_)[:20]):  # Show first 20
                    print(f"{i+1:2}. {disease}")
                if len(model.classes_) > 20:
                    print(f"... and {len(model.classes_) - 20} more")
            
            elif choice == '4':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice!")
    
    except FileNotFoundError:
        print("Model file 'medical_triage_model.pkl' not found!")
        print("Please train the model first by running:")
        print("python models/medical_triage_model_fixed.py")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    load_and_test_model()