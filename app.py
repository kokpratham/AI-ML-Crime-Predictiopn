# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

class CrimePredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.crime_type_encoder = LabelEncoder()
        self.is_trained = False
        
    def generate_synthetic_data(self, n_samples=5000):
        """Generate synthetic crime data for demonstration"""
        np.random.seed(42)
        
        # Generate features
        data = {
            'hour': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'month': np.random.randint(1, 13, n_samples),
            'temperature': np.random.normal(20, 10, n_samples),
            'unemployment_rate': np.random.normal(5, 2, n_samples),
            'population_density': np.random.normal(1000, 500, n_samples),
            'police_stations_nearby': np.random.randint(0, 5, n_samples),
            'previous_crimes_area': np.random.poisson(3, n_samples),
            'lighting_quality': np.random.randint(1, 6, n_samples),  # 1-5 scale
            'economic_index': np.random.normal(50, 15, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create crime probability based on features
        crime_prob = (
            0.3 * (df['hour'].apply(lambda x: 1 if x >= 22 or x <= 4 else 0)) +
            0.2 * (df['unemployment_rate'] > 7).astype(int) +
            0.15 * (df['lighting_quality'] <= 2).astype(int) +
            0.1 * (df['police_stations_nearby'] == 0).astype(int) +
            0.1 * (df['previous_crimes_area'] > 5).astype(int) +
            0.15 * np.random.random(n_samples)
        )
        
        # Generate crime types and binary target
        df['crime_occurred'] = (crime_prob > 0.4).astype(int)
        
        crime_types = ['Theft', 'Assault', 'Burglary', 'Vandalism', 'Drug_Related', 'No_Crime']
        crime_weights = np.where(df['crime_occurred'] == 1, 
                               [0.3, 0.25, 0.2, 0.15, 0.1, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        
        df['crime_type'] = np.random.choice(crime_types, n_samples, 
                                          p=[0.15, 0.12, 0.1, 0.08, 0.05, 0.5])
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        # Create features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 4)).astype(int)
        df['is_winter'] = ((df['month'] <= 2) | (df['month'] == 12)).astype(int)
        df['high_crime_area'] = (df['previous_crimes_area'] > df['previous_crimes_area'].median()).astype(int)
        
        return df
    
    def train_model(self, df=None):
        """Train the crime prediction model"""
        if df is None:
            df = self.generate_synthetic_data()
            
        df = self.preprocess_data(df)
        
        # Features for prediction
        feature_columns = [
            'hour', 'day_of_week', 'month', 'temperature', 'unemployment_rate',
            'population_density', 'police_stations_nearby', 'previous_crimes_area',
            'lighting_quality', 'economic_index', 'is_weekend', 'is_night',
            'is_winter', 'high_crime_area'
        ]
        
        X = df[feature_columns]
        y = df['crime_occurred']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        return {
            'accuracy': accuracy,
            'feature_importance': dict(zip(feature_columns, self.model.feature_importances_))
        }
    
    def predict_crime_risk(self, input_data):
        """Predict crime risk for given input"""
        if not self.is_trained:
            return {'error': 'Model not trained yet'}
        
        # Preprocess input
        df = pd.DataFrame([input_data])
        df = self.preprocess_data(df)
        
        feature_columns = [
            'hour', 'day_of_week', 'month', 'temperature', 'unemployment_rate',
            'population_density', 'police_stations_nearby', 'previous_crimes_area',
            'lighting_quality', 'economic_index', 'is_weekend', 'is_night',
            'is_winter', 'high_crime_area'
        ]
        
        X = df[feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # Predict
        crime_probability = self.model.predict_proba(X_scaled)[0]
        crime_prediction = self.model.predict(X_scaled)[0]
        
        risk_level = 'Low'
        if crime_probability[1] > 0.7:
            risk_level = 'High'
        elif crime_probability[1] > 0.4:
            risk_level = 'Medium'
        
        return {
            'crime_probability': float(crime_probability[1]),
            'crime_prediction': int(crime_prediction),
            'risk_level': risk_level,
            'confidence': float(max(crime_probability))
        }
    
    def save_model(self):
        """Save the trained model"""
        if not os.path.exists('models'):
            os.makedirs('models')
            
        with open('models/crime_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_model(self):
        """Load the trained model"""
        try:
            with open('models/crime_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            with open('models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_trained = True
            return True
        except FileNotFoundError:
            return False

# Initialize the model
crime_model = CrimePredictionModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    """Train the model endpoint"""
    try:
        results = crime_model.train_model()
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'accuracy': results['accuracy'],
            'feature_importance': results['feature_importance']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Training failed: {str(e)}'
        })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        data = request.json
        
        # Validate input data
        required_fields = [
            'hour', 'day_of_week', 'month', 'temperature', 'unemployment_rate',
            'population_density', 'police_stations_nearby', 'previous_crimes_area',
            'lighting_quality', 'economic_index'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'message': f'Missing required field: {field}'
                })
        
        # Make prediction
        result = crime_model.predict_crime_risk(data)
        
        if 'error' in result:
            return jsonify({
                'success': False,
                'message': result['error']
            })
        
        return jsonify({
            'success': True,
            'prediction': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Prediction failed: {str(e)}'
        })

@app.route('/model_status')
def model_status():
    """Check if model is trained"""
    return jsonify({
        'is_trained': crime_model.is_trained,
        'model_exists': os.path.exists('models/crime_model.pkl')
    })

if __name__ == '__main__':
    # Try to load existing model
    if crime_model.load_model():
        print("Loaded existing model")
    else:
        print("No existing model found. Please train the model first.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)