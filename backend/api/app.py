# app.py - Main Flask API for SAR Narrative Generator

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow frontend to connect

# Configuration
class Config:
    BASE_DIR = Path(__file__).parent.parent.absolute()
    DATA_DIR = BASE_DIR / 'data'
    SYNTHETIC_DIR = DATA_DIR / 'synthetic'
    MODELS_DIR = BASE_DIR / 'models'
    
config = Config()

# Load balanced dataset for reference
def load_balanced_data():
    """Load the balanced dataset created by CTGAN"""
    balanced_path = config.SYNTHETIC_DIR / 'balanced_training_data.csv'
    if balanced_path.exists():
        return pd.read_csv(balanced_path)
    return None

# Simple rule-based model for demonstration
# (In production, you'd load a trained ML model)
class SimpleFraudDetector:
    def __init__(self):
        self.threshold = 0.7
        self.feature_importance = {
            'amount': 0.35,
            'time_pattern': 0.25,
            'velocity': 0.30,
            'merchant_risk': 0.10
        }
    
    def predict(self, transaction_data):
        """
        Simple rule-based fraud detection
        In real implementation, this would use your trained model
        """
        risk_score = 0.0
        reasons = []
        
        # Rule 1: Check amount
        amount = transaction_data.get('amount', 0)
        if amount > 100000:
            risk_score += 0.4
            reasons.append("High amount transaction (>₹100,000)")
        elif amount > 50000:
            risk_score += 0.2
            reasons.append("Moderate amount transaction (>₹50,000)")
        
        # Rule 2: Check transaction count (velocity)
        tx_count = transaction_data.get('transaction_count_7d', 1)
        if tx_count > 47:
            risk_score += 0.3
            reasons.append(f"High velocity: {tx_count} transactions in 7 days")
        elif tx_count > 20:
            risk_score += 0.15
            reasons.append(f"Moderate velocity: {tx_count} transactions in 7 days")
        
        # Rule 3: Check unique counterparties
        unique_cp = transaction_data.get('unique_counterparties_7d', 1)
        if unique_cp > 47:
            risk_score += 0.3
            reasons.append(f"Many unique counterparties: {unique_cp} in 7 days")
        
        # Rule 4: International transaction
        is_international = transaction_data.get('is_international', False)
        if is_international:
            risk_score += 0.15
            reasons.append("International transaction")
        
        # Cap risk score at 1.0
        risk_score = min(risk_score, 1.0)
        
        # Determine if suspicious
        is_suspicious = risk_score >= self.threshold
        
        return {
            'risk_score': round(risk_score, 3),
            'is_suspicious': is_suspicious,
            'reasons': reasons,
            'threshold_used': self.threshold
        }

# Initialize detector
detector = SimpleFraudDetector()

# Routes
@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        'name': 'Barkley SAR Narrative Generator API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            '/': 'This information',
            '/api/health': 'Health check',
            '/api/predict': 'POST - Predict transaction risk',
            '/api/synthetic-examples': 'GET - View synthetic fraud examples',
            '/api/generate-sar': 'POST - Generate SAR narrative',
            '/api/threshold': 'GET/POST - View/update detection threshold'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Check if balanced data exists
    balanced_data = load_balanced_data()
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'data_loaded': balanced_data is not None,
        'model_ready': True,
        'threshold': detector.threshold
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict if a transaction is suspicious
    Expects JSON with transaction data
    """
    try:
        # Get JSON data from request
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Log received data (for audit trail)
        print(f"Received prediction request: {json.dumps(data, indent=2)}")
        
        # Get prediction from detector
        prediction = detector.predict(data)
        
        # Add metadata
        response = {
            'success': True,
            'prediction': prediction,
            'input_summary': {
                'amount': data.get('amount', 'N/A'),
                'customer_id': data.get('customer_id', 'N/A'),
                'tx_count_7d': data.get('transaction_count_7d', 'N/A'),
                'unique_cp_7d': data.get('unique_counterparties_7d', 'N/A')
            },
            'timestamp': datetime.now().isoformat(),
            'model_version': 'rule-based-v1'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/synthetic-examples', methods=['GET'])
def get_synthetic_examples():
    """Return examples of synthetic fraud data"""
    try:
        # Try to load latest synthetic fraud
        latest_path = config.SYNTHETIC_DIR / 'fraud_samples_latest.csv'
        
        if latest_path.exists():
            df = pd.read_csv(latest_path)
            # Return first 10 examples
            examples = df.head(10).to_dict('records')
            
            # Get statistics
            stats = {
                'total_samples': len(df),
                'avg_amount': float(df['Amount'].mean()),
                'max_amount': float(df['Amount'].max()),
                'min_amount': float(df['Amount'].min())
            }
            
            return jsonify({
                'success': True,
                'count': len(examples),
                'examples': examples,
                'statistics': stats
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No synthetic data found. Run CTGAN first.'
            }), 404
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/generate-sar', methods=['POST'])
def generate_sar():
    """
    Generate a SAR narrative based on transaction data
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # First, get prediction
        prediction = detector.predict(data)
        
        # Generate SAR narrative based on prediction
        risk_level = "HIGH" if prediction['is_suspicious'] else "LOW"
        
        # Current date for SAR
        current_date = datetime.now().strftime('%Y-%m-%d')
        case_id = f"SAR-{datetime.now().strftime('%Y%m%d')}-{np.random.randint(1000, 9999)}"
        
        # Build narrative
        narrative = f"""
SUSPICIOUS ACTIVITY REPORT - DRAFT
===================================
Case ID: {case_id}
Date: {current_date}
Risk Level: {risk_level} (Score: {prediction['risk_score']})

SUBJECT INFORMATION:
-------------------
Customer ID: {data.get('customer_id', 'Unknown')}
Account Type: {data.get('account_type', 'Savings')}
Customer Tenure: {data.get('customer_tenure', 'Unknown')} years

TRANSACTION SUMMARY:
-------------------
• Transaction Amount: ₹{data.get('amount', 0):,.2f}
• Transactions (7 days): {data.get('transaction_count_7d', 0)}
• Unique Counterparties: {data.get('unique_counterparties_7d', 0)}
• Alert Reason: {data.get('alert_reason', 'Rule-based detection')}

SUSPICIOUS PATTERNS DETECTED:
----------------------------
{chr(10).join(['• ' + reason for reason in prediction['reasons']])}

NARRATIVE:
---------
The subject account demonstrated activity consistent with potential 
money laundering or fraud. {prediction['reasons'][0] if prediction['reasons'] else 'The transaction pattern warrants review.'}

The transaction velocity of {data.get('transaction_count_7d', 0)} transactions 
to {data.get('unique_counterparties_7d', 0)} unique counterparties within a 
7-day period is unusual compared to the customer's historical pattern.

Based on the analysis, this activity requires further investigation 
and potential filing of a complete SAR with appropriate authorities.

AUDIT TRAIL:
-----------
• Detection Method: Rule-based heuristic model
• Threshold Used: {detector.threshold}
• Feature Importance: Amount({detector.feature_importance['amount']}), 
  Velocity({detector.feature_importance['velocity']})
• Model Version: rule-based-v1

---
DRAFT - Requires Analyst Review and Approval
This document is preliminary and subject to verification.
"""
        
        # Prepare audit trail
        audit_trail = {
            'case_id': case_id,
            'timestamp': datetime.now().isoformat(),
            'input_data': data,
            'model_output': prediction,
            'feature_importance': detector.feature_importance,
            'model_version': 'rule-based-v1'
        }
        
        return jsonify({
            'success': True,
            'case_id': case_id,
            'risk_level': risk_level,
            'risk_score': prediction['risk_score'],
            'narrative': narrative,
            'audit_trail': audit_trail,
            'requires_review': prediction['is_suspicious']
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/threshold', methods=['GET', 'POST'])
def handle_threshold():
    """Get or update detection threshold"""
    global detector
    
    if request.method == 'GET':
        return jsonify({
            'current_threshold': detector.threshold,
            'min': 0.1,
            'max': 0.95,
            'default': 0.7
        })
    
    elif request.method == 'POST':
        try:
            data = request.json
            new_threshold = float(data.get('threshold', 0.7))
            
            # Validate
            if 0.0 <= new_threshold <= 1.0:
                detector.threshold = new_threshold
                return jsonify({
                    'success': True,
                    'message': f'Threshold updated to {new_threshold}',
                    'current_threshold': detector.threshold
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Threshold must be between 0 and 1'
                }), 400
        
        except Exception as e:
            return jsonify({'error': str(e), 'success': False}), 500

# Run the app
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 BARKLEY SAR NARRATIVE GENERATOR API")
    print("=" * 60)
    print(f"\n📁 Data directory: {config.DATA_DIR}")
    print(f"📁 Synthetic data: {config.SYNTHETIC_DIR}")
    print(f"\n📍 API will be available at: http://127.0.0.1:5000")
    print(f"📍 Health check: http://127.0.0.1:5000/api/health")
    print(f"\n📋 Available endpoints:")
    print("   • GET  /                  - API info")
    print("   • GET  /api/health        - Health check")
    print("   • POST /api/predict       - Predict transaction")
    print("   • GET  /api/synthetic-examples - View synthetic data")
    print("   • POST /api/generate-sar  - Generate SAR narrative")
    print("   • GET/POST /api/threshold - View/update threshold")
    print("\n⏸️  Press Ctrl+C to stop the server")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)