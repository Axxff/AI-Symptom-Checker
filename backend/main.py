from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Medical Triage Assistant API",
    description="AI-powered medical triage system for symptom assessment and urgency classification",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
ml_model = None
model_data = None
symptom_columns = None

# Triage Rules Engine
class TriageRulesEngine:
    def __init__(self):
        self.emergency_keywords = [
            'chest pain', 'difficulty breathing', 'breathlessness', 'unconscious', 
            'seizure', 'severe bleeding', 'stroke', 'heart attack', 'severe abdominal pain', 
            'high fever', 'severe headache', 'paralysis', 'brain hemorrhage'
        ]
        
        self.urgent_keywords = [
            'persistent pain', 'moderate fever', 'vomiting', 'diarrhoea',
            'skin rash', 'joint pain', 'back pain', 'stomach pain', 'fatigue'
        ]
        
        self.routine_keywords = [
            'mild headache', 'common cold', 'minor pain', 'mild fever', 'acne'
        ]
        
        self.combination_rules = [
            {
                'symptoms': ['chest_pain', 'breathlessness'],
                'severity': 'emergency',
                'message': 'Chest pain with breathing difficulty requires immediate medical attention'
            },
            {
                'symptoms': ['high_fever', 'headache', 'stiff_neck'],
                'severity': 'emergency',
                'message': 'High fever with headache and neck stiffness - possible serious infection'
            },
            {
                'symptoms': ['abdominal_pain', 'vomiting', 'high_fever'],
                'severity': 'urgent',
                'message': 'Multiple symptoms may indicate serious condition - seek medical care'
            }
        ]
    
    def classify_urgency(self, symptoms_text: str, predicted_condition: str) -> Dict:
        """Classify urgency based on symptoms and predicted condition"""
        normalized_symptoms = symptoms_text.lower().strip()
        
        # Check combination rules first
        for rule in self.combination_rules:
            symptoms_present = all(
                symptom.replace('_', ' ') in normalized_symptoms 
                for symptom in rule['symptoms']
            )
            
            if symptoms_present:
                return {
                    'urgency_level': rule['severity'],
                    'urgency_score': self.get_urgency_score(rule['severity']),
                    'message': rule['message'],
                    'rule_triggered': True,
                    'color': self.get_urgency_color(rule['severity'])
                }
        
        # Check individual emergency keywords
        for keyword in self.emergency_keywords:
            if keyword in normalized_symptoms:
                return {
                    'urgency_level': 'emergency',
                    'urgency_score': 9,
                    'message': f'Emergency condition detected: {keyword.title()}',
                    'rule_triggered': True,
                    'color': 'red'
                }
        
        # Check urgent keywords
        for keyword in self.urgent_keywords:
            if keyword in normalized_symptoms:
                return {
                    'urgency_level': 'urgent',
                    'urgency_score': 6,
                    'message': f'Urgent condition detected: {keyword.title()}',
                    'rule_triggered': True,
                    'color': 'yellow'
                }
        
        # Check routine keywords
        for keyword in self.routine_keywords:
            if keyword in normalized_symptoms:
                return {
                    'urgency_level': 'routine',
                    'urgency_score': 3,
                    'message': f'Routine condition detected: {keyword.title()}',
                    'rule_triggered': True,
                    'color': 'green'
                }
        
        # Condition-based urgency (backup)
        condition_urgency = self.get_condition_urgency(predicted_condition)
        return {
            'urgency_level': condition_urgency['level'],
            'urgency_score': condition_urgency['score'],
            'message': condition_urgency['message'],
            'rule_triggered': False,
            'color': self.get_urgency_color(condition_urgency['level'])
        }
    
    def get_condition_urgency(self, condition: str) -> Dict:
        """Get urgency based on predicted condition"""
        condition_lower = condition.lower()
        
        # Emergency conditions
        if any(term in condition_lower for term in ['heart attack', 'stroke', 'pneumonia', 'hepatitis']):
            return {'level': 'emergency', 'score': 8, 'message': f'{condition} may require immediate attention'}
        
        # Urgent conditions  
        if any(term in condition_lower for term in ['diabetes', 'hypertension', 'malaria', 'typhoid']):
            return {'level': 'urgent', 'score': 6, 'message': f'{condition} should be evaluated by a healthcare provider soon'}
        
        # Routine conditions
        if any(term in condition_lower for term in ['common cold', 'allergy', 'acne', 'migraine']):
            return {'level': 'routine', 'score': 3, 'message': f'{condition} can typically be managed with routine care'}
        
        # Default to urgent if unsure
        return {'level': 'urgent', 'score': 5, 'message': f'{condition} should be evaluated by a healthcare provider'}
    
    def get_urgency_score(self, urgency_level: str) -> int:
        urgency_scores = {'routine': 3, 'urgent': 6, 'emergency': 9}
        return urgency_scores.get(urgency_level, 5)
    
    def get_urgency_color(self, urgency_level: str) -> str:
        color_mapping = {'routine': 'green', 'urgent': 'yellow', 'emergency': 'red'}
        return color_mapping.get(urgency_level, 'gray')

rules_engine = TriageRulesEngine()

# Pydantic models
class SymptomRequest(BaseModel):
    symptoms: str
    patient_age: Optional[int] = None
    patient_gender: Optional[str] = None

class PredictionResponse(BaseModel):
    conditions: List[Dict]
    urgency: Dict
    ml_prediction: Dict
    disclaimer: str
    timestamp: str

def load_model():
    """Load the trained model at startup"""
    global ml_model, model_data, symptom_columns
    
    model_files = [
        'improved_medical_triage_model.pkl',
        'medical_triage_model.pkl'
    ]
    
    for model_file in model_files:
        try:
            model_data = joblib.load(model_file)
            ml_model = model_data['model']
            symptom_columns = model_data['symptom_columns']
            print(f"Successfully loaded model from {model_file}")
            return True
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error loading {model_file}: {e}")
            continue
    
    print("No model file found! Please train a model first.")
    return False

def predict_from_text(symptom_text: str) -> Dict:
    """Make prediction from symptom text"""
    if not ml_model or not symptom_columns:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    symptom_text = symptom_text.lower()
    
    # Create base feature vector  
    base_symptom_cols = [col for col in symptom_columns 
                        if col not in ['total_symptoms', 'pain_score', 'fever_score', 'digestive_score', 'respiratory_score']]
    
    feature_vector = np.zeros(len(base_symptom_cols))
    
    # Map symptoms from text
    for i, symptom_col in enumerate(base_symptom_cols):
        readable_symptom = symptom_col.replace('_', ' ')
        if readable_symptom in symptom_text or symptom_col in symptom_text:
            feature_vector[i] = 1
    
    # If we have engineered features, calculate them
    if len(symptom_columns) > len(base_symptom_cols):
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
        
        # Scale features if scaler is available
        if 'scaler' in model_data and model_data['scaler']:
            full_feature_vector = model_data['scaler'].transform(full_feature_vector.reshape(1, -1))[0]
    else:
        full_feature_vector = feature_vector
    
    # Reshape for prediction
    full_feature_vector = full_feature_vector.reshape(1, -1)
    
    # Get prediction
    prediction = ml_model.predict(full_feature_vector)[0]
    probabilities = ml_model.predict_proba(full_feature_vector)[0]
    
    # Get top 3 predictions
    top_indices = np.argsort(probabilities)[-3:][::-1]
    top_conditions = [ml_model.classes_[i] for i in top_indices]
    top_probabilities = [probabilities[i] for i in top_indices]
    
    return {
        'top_prediction': prediction,
        'top_conditions': top_conditions,
        'probabilities': top_probabilities,
        'symptoms_detected': int(np.sum(feature_vector))
    }

@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    success = load_model()
    if not success:
        print("Warning: API started without a trained model!")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Medical Triage Assistant API",
        "status": "running",
        "model_loaded": ml_model is not None,
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict - POST request to get condition predictions",
            "health": "/health - Check API health status", 
            "symptoms": "/symptoms - Get list of available symptoms",
            "docs": "/docs - Interactive API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": ml_model is not None,
        "timestamp": datetime.now().isoformat(),
        "available_symptoms": len(symptom_columns) if symptom_columns else 0
    }

@app.get("/symptoms")
async def get_symptoms():
    """Get list of available symptoms"""
    if not symptom_columns:
        raise HTTPException(status_code=503, detail="Model not loaded - symptoms not available")
    
    # Filter out engineered features and return readable symptoms
    base_symptoms = [col for col in symptom_columns 
                    if col not in ['total_symptoms', 'pain_score', 'fever_score', 'digestive_score', 'respiratory_score']]
    
    readable_symptoms = [symptom.replace('_', ' ').title() for symptom in base_symptoms]
    
    return {
        "symptoms": readable_symptoms,
        "count": len(readable_symptoms),
        "categories": {
            "pain": [s for s in readable_symptoms if 'Pain' in s],
            "fever": [s for s in readable_symptoms if 'Fever' in s],
            "digestive": [s for s in readable_symptoms if any(word in s.lower() for word in ['stomach', 'nausea', 'vomit', 'diarrhoea', 'constipation'])],
            "respiratory": [s for s in readable_symptoms if any(word in s.lower() for word in ['cough', 'breathing', 'chest', 'congestion'])]
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_conditions(request: SymptomRequest):
    """Main prediction endpoint"""
    try:
        symptoms_text = request.symptoms.strip()
        
        if not symptoms_text:
            raise HTTPException(status_code=400, detail="Symptoms text cannot be empty")
        
        if not ml_model:
            raise HTTPException(status_code=503, detail="ML model not loaded")
        
        # Get ML model predictions
        ml_result = predict_from_text(symptoms_text)
        
        # Get urgency classification from rules engine
        urgency_result = rules_engine.classify_urgency(symptoms_text, ml_result['top_prediction'])
        
        # Format conditions for response
        conditions = []
        for i, condition in enumerate(ml_result['top_conditions']):
            conditions.append({
                "condition": condition.replace('_', ' ').title().strip(),
                "probability": round(ml_result['probabilities'][i] * 100, 1),
                "rank": i + 1,
                "confidence_level": "high" if ml_result['probabilities'][i] > 0.7 else "medium" if ml_result['probabilities'][i] > 0.4 else "low"
            })
        
        # Prepare response
        response = {
            "conditions": conditions,
            "urgency": urgency_result,
            "ml_prediction": {
                "top_condition": ml_result['top_prediction'].replace('_', ' ').title().strip(),
                "confidence": round(max(ml_result['probabilities']) * 100, 1),
                "symptoms_detected": ml_result['symptoms_detected'],
                "total_possible_symptoms": len(symptom_columns) if symptom_columns else 0
            },
            "disclaimer": "⚠️ IMPORTANT: This is an AI-powered preliminary assessment and should NOT replace professional medical advice. Always consult with a qualified healthcare provider for accurate diagnosis and treatment. In case of emergency, seek immediate medical attention or call emergency services.",
            "timestamp": datetime.now().isoformat()
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/predict/test")
async def test_prediction():
    """Test endpoint with sample predictions"""
    if not ml_model:
        return {"error": "Model not loaded"}
    
    test_cases = [
        "headache and fever",
        "chest pain and breathlessness",
        "stomach pain and nausea", 
        "cough and fatigue",
        "skin rash and itching"
    ]
    
    results = []
    for symptoms in test_cases:
        try:
            ml_result = predict_from_text(symptoms)
            urgency_result = rules_engine.classify_urgency(symptoms, ml_result['top_prediction'])
            
            results.append({
                "symptoms": symptoms,
                "prediction": ml_result['top_prediction'],
                "confidence": f"{max(ml_result['probabilities']) * 100:.1f}%",
                "urgency": urgency_result['urgency_level'],
                "urgency_score": urgency_result['urgency_score']
            })
        except Exception as e:
            results.append({
                "symptoms": symptoms,
                "error": str(e)
            })
    
    return {"test_results": results}

@app.get("/stats")
async def get_model_stats():
    """Get model statistics and information"""
    if not ml_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    stats = {
        "model_type": type(ml_model).__name__,
        "total_conditions": len(ml_model.classes_),
        "conditions": sorted(ml_model.classes_),
        "total_symptoms": len(symptom_columns) if symptom_columns else 0,
        "model_loaded": True
    }
    
    # Add feature importance if available
    if hasattr(ml_model, 'feature_importances_') and 'feature_importance' in model_data:
        importance_data = model_data.get('feature_importance', [])
        if len(importance_data) > 0:
            # Get top 10 most important features
            feature_importance_pairs = list(zip(symptom_columns, importance_data))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            stats['top_important_symptoms'] = [
                {
                    "symptom": pair[0].replace('_', ' ').title(),
                    "importance": round(pair[1], 4)
                }
                for pair in feature_importance_pairs[:10]
            ]
    
    return stats

if __name__ == "__main__":
    import uvicorn
    
    # Load model before starting server
    load_model()
    
    # Start server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=False,  # Set to True for development
        log_level="info"
    )