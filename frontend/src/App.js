import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [symptoms, setSymptoms] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [availableSymptoms, setAvailableSymptoms] = useState([]);
  const [apiHealth, setApiHealth] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Check API health on component mount
  useEffect(() => {
    checkApiHealth();
    fetchAvailableSymptoms();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`);
      setApiHealth(response.data);
    } catch (err) {
      console.error('API health check failed:', err);
      setApiHealth({ status: 'offline', model_loaded: false });
    }
  };

  const fetchAvailableSymptoms = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/symptoms`);
      setAvailableSymptoms(response.data.symptoms || []);
    } catch (err) {
      console.error('Failed to fetch symptoms:', err);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!symptoms.trim()) {
      setError('Please enter your symptoms');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, {
        symptoms: symptoms
      });
      
      setPrediction(response.data);
    } catch (err) {
      if (err.response?.status === 503) {
        setError('Model not loaded on server. Please ensure the ML model is trained and loaded.');
      } else {
        setError('Failed to get prediction. Please try again.');
      }
      console.error('API Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const getUrgencyStyle = (urgencyLevel) => {
    const styles = {
      emergency: { 
        backgroundColor: '#fee2e2', 
        borderColor: '#dc2626', 
        color: '#dc2626',
        borderWidth: '2px',
        boxShadow: '0 4px 12px rgba(220, 38, 38, 0.2)'
      },
      urgent: { 
        backgroundColor: '#fef3c7', 
        borderColor: '#d97706', 
        color: '#d97706',
        borderWidth: '2px',
        boxShadow: '0 4px 12px rgba(217, 119, 6, 0.2)'
      },
      routine: { 
        backgroundColor: '#dcfce7', 
        borderColor: '#16a34a', 
        color: '#16a34a',
        borderWidth: '2px',
        boxShadow: '0 4px 12px rgba(22, 163, 74, 0.2)'
      },
      unknown: { 
        backgroundColor: '#f3f4f6', 
        borderColor: '#6b7280', 
        color: '#6b7280',
        borderWidth: '1px'
      }
    };
    return styles[urgencyLevel] || styles.unknown;
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 70) return '#16a34a'; // Green
    if (confidence >= 40) return '#d97706'; // Orange
    return '#dc2626'; // Red
  };

  const addSymptomToInput = (symptom) => {
    const currentSymptoms = symptoms.toLowerCase();
    const symptomLower = symptom.toLowerCase();
    
    if (!currentSymptoms.includes(symptomLower)) {
      const newSymptoms = symptoms ? `${symptoms}, ${symptom.toLowerCase()}` : symptom.toLowerCase();
      setSymptoms(newSymptoms);
    }
  };

  const testWithSample = (sampleSymptoms) => {
    setSymptoms(sampleSymptoms);
    setPrediction(null);
    setError('');
  };

  return (
    <div className="App">
      <header className="app-header">
        <div className="container">
          <div className="header-content">
            <div className="header-icon">🏥</div>
            <h1>Medical Triage Assistant</h1>
            <p>AI-powered symptom assessment and urgency classification</p>
            
            {/* API Status Indicator */}
            <div className="api-status">
              <span className={`status-indicator ${apiHealth?.status === 'healthy' ? 'online' : 'offline'}`}></span>
              {apiHealth?.model_loaded ? 'Model Ready' : 'Model Loading...'}
            </div>
          </div>
        </div>
      </header>

      <main className="main-content">
        <div className="container">
          {/* Quick Test Buttons */}
          <div className="quick-tests">
            <h3>Try Sample Symptoms:</h3>
            <div className="test-buttons">
              {[
                "headache and fever",
                "chest pain and breathlessness", 
                "stomach pain and nausea",
                "cough and fatigue",
                "skin rash and itching"
              ].map((sample, index) => (
                <button 
                  key={index}
                  className="test-btn"
                  onClick={() => testWithSample(sample)}
                >
                  {sample}
                </button>
              ))}
            </div>
          </div>

          <div className="input-section">
            <form onSubmit={handleSubmit} className="symptom-form">
              <div className="form-group">
                <label htmlFor="symptoms">
                  👤 Describe your symptoms
                </label>
                <textarea
                  id="symptoms"
                  value={symptoms}
                  onChange={(e) => setSymptoms(e.target.value)}
                  placeholder="e.g., headache, fever, chest pain, nausea..."
                  rows="4"
                  className="symptom-input"
                />
                <div className="input-helper">
                  💡 Tip: Use simple terms like "headache", "fever", "chest pain", etc.
                </div>
              </div>
              
              <button 
                type="submit" 
                disabled={loading || !apiHealth?.model_loaded}
                className={`submit-btn ${loading ? 'loading' : ''}`}
              >
                {loading ? '🔍 Analyzing...' : '🔬 Analyze Symptoms'}
              </button>
            </form>

            {/* Advanced Options */}
            <div className="advanced-section">
              <button 
                className="toggle-advanced"
                onClick={() => setShowAdvanced(!showAdvanced)}
              >
                {showAdvanced ? '▼' : '▶'} Advanced Options
              </button>
              
              {showAdvanced && (
                <div className="advanced-content">
                  <h4>Available Symptoms ({availableSymptoms.length}):</h4>
                  <div className="symptoms-grid">
                    {availableSymptoms.slice(0, 20).map((symptom, index) => (
                      <button
                        key={index}
                        className="symptom-tag"
                        onClick={() => addSymptomToInput(symptom)}
                      >
                        + {symptom}
                      </button>
                    ))}
                  </div>
                  {availableSymptoms.length > 20 && (
                    <p className="symptoms-note">
                      ... and {availableSymptoms.length - 20} more symptoms available
                    </p>
                  )}
                </div>
              )}
            </div>

            {error && (
              <div className="error-message">
                <span className="error-icon">⚠️</span>
                {error}
              </div>
            )}
          </div>

          {prediction && (
            <div className="results-section">
              {/* Urgency Assessment */}
              <div className="urgency-card" style={getUrgencyStyle(prediction.urgency.urgency_level)}>
                <h3>🚨 Urgency Assessment</h3>
                <div className="urgency-level">
                  {prediction.urgency.urgency_level.toUpperCase()}
                </div>
                <p className="urgency-message">{prediction.urgency.message}</p>
                <div className="urgency-details">
                  <span>Score: {prediction.urgency.urgency_score}/10</span>
                  {prediction.urgency.rule_triggered && (
                    <span className="rule-badge">Rule-based</span>
                  )}
                </div>
              </div>

              {/* ML Prediction Summary */}
              <div className="ml-summary-card">
                <h3>🤖 AI Analysis</h3>
                <div className="ml-summary">
                  <div className="primary-prediction">
                    <span className="prediction-label">Most Likely:</span>
                    <span className="prediction-value">{prediction.ml_prediction.top_condition}</span>
                    <span 
                      className="confidence-badge"
                      style={{ backgroundColor: getConfidenceColor(prediction.ml_prediction.confidence) }}
                    >
                      {prediction.ml_prediction.confidence}%
                    </span>
                  </div>
                  <div className="detection-stats">
                    <span>Symptoms detected: {prediction.ml_prediction.symptoms_detected}</span>
                    <span>Total possible: {prediction.ml_prediction.total_possible_symptoms}</span>
                  </div>
                </div>
              </div>

              {/* Detailed Predictions */}
              <div className="predictions-card">
                <h3>📋 Possible Conditions</h3>
                <div className="conditions-list">
                  {prediction.conditions.map((condition, index) => (
                    <div key={index} className="condition-item">
                      <div className="condition-header">
                        <span className="condition-rank">#{condition.rank}</span>
                        <span className="condition-name">{condition.condition}</span>
                        <span className={`confidence-level ${condition.confidence_level}`}>
                          {condition.confidence_level} confidence
                        </span>
                      </div>
                      <div className="probability-bar-container">
                        <div className="probability-bar">
                          <div 
                            className="probability-fill"
                            style={{ 
                              width: `${condition.probability}%`,
                              backgroundColor: getConfidenceColor(condition.probability)
                            }}
                          ></div>
                        </div>
                        <span className="probability-text">{condition.probability}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Timestamp */}
              <div className="prediction-meta">
                <small>
                  Analysis completed: {new Date(prediction.timestamp).toLocaleString()}
                </small>
              </div>

              {/* Important Disclaimer */}
              <div className="disclaimer">
                <div className="disclaimer-icon">⚠️</div>
                <div className="disclaimer-content">
                  <h4>Important Medical Disclaimer</h4>
                  <p>{prediction.disclaimer}</p>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <div className="container">
          <p>Medical Triage Assistant v1.0 | For educational purposes only</p>
          <p>Always consult healthcare professionals for medical advice</p>
        </div>
      </footer>
    </div>
  );
}

export default App;