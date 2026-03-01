# app_with_db.py - Flask API with PostgreSQL audit trail
# COMPLETE VERSION WITH ALL ENDPOINTS

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime
import psycopg2
from psycopg2.extras import Json, RealDictCursor
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
class Config:
    BASE_DIR = Path(__file__).parent.parent.absolute()
    DATA_DIR = BASE_DIR / 'data'
    SYNTHETIC_DIR = DATA_DIR / 'synthetic'
    MODELS_DIR = BASE_DIR / 'models'
    
    # Database configuration - CHANGE PASSWORD TO YOUR POSTGRES PASSWORD!
    DB_CONFIG = {
        "host": "localhost",
        "database": "barkley_sar",
        "user": "postgres",
        "password": "postgres",  # CHANGE THIS to your password
        "port": "5432"
    }
    
config = Config()

# Database connection helper
def get_db_connection():
    """Create a connection to PostgreSQL"""
    return psycopg2.connect(**config.DB_CONFIG)

# Initialize database table if it doesn't exist
def init_database():
    """Ensure the audit_log table exists"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'audit_log'
            );
        """)
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            print("📦 Creating audit_log table...")
            cur.execute("""
                CREATE TABLE audit_log (
                    id SERIAL PRIMARY KEY,
                    case_id VARCHAR(50),
                    input_data JSONB,
                    model_used VARCHAR(100),
                    feature_importance JSONB,
                    llm_prompt TEXT,
                    llm_response TEXT,
                    reasoning_trace JSONB,
                    analyst_edits TEXT,
                    final_sar TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    approved_at TIMESTAMP,
                    created_by VARCHAR(100),
                    approved_by VARCHAR(100)
                );
                
                CREATE INDEX idx_audit_log_case_id ON audit_log(case_id);
                CREATE INDEX idx_audit_log_created_at ON audit_log(created_at);
            """)
            conn.commit()
            print("✅ audit_log table created")
        else:
            print("✅ audit_log table already exists")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"⚠️ Database init warning: {e}")

# Initialize database on startup
init_database()

# Simple rule-based model
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
        risk_score = 0.0
        reasons = []
        
        amount = transaction_data.get('amount', 0)
        if amount > 100000:
            risk_score += 0.4
            reasons.append("High amount transaction (>₹100,000)")
        elif amount > 50000:
            risk_score += 0.2
            reasons.append("Moderate amount transaction (>₹50,000)")
        
        tx_count = transaction_data.get('transaction_count_7d', 1)
        if tx_count > 47:
            risk_score += 0.3
            reasons.append(f"High velocity: {tx_count} transactions in 7 days")
        elif tx_count > 20:
            risk_score += 0.15
            reasons.append(f"Moderate velocity: {tx_count} transactions in 7 days")
        
        unique_cp = transaction_data.get('unique_counterparties_7d', 1)
        if unique_cp > 47:
            risk_score += 0.3
            reasons.append(f"Many unique counterparties: {unique_cp} in 7 days")
        
        is_international = transaction_data.get('is_international', False)
        if is_international:
            risk_score += 0.15
            reasons.append("International transaction")
        
        risk_score = min(risk_score, 1.0)
        is_suspicious = risk_score >= self.threshold
        
        return {
            'risk_score': round(risk_score, 3),
            'is_suspicious': is_suspicious,
            'reasons': reasons,
            'threshold_used': self.threshold
        }

detector = SimpleFraudDetector()

# Load synthetic data for examples
def load_synthetic_examples():
    try:
        latest_path = config.SYNTHETIC_DIR / 'fraud_samples_latest.csv'
        if latest_path.exists():
            return pd.read_csv(latest_path)
        return None
    except:
        return None

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'name': 'Barkley SAR Generator API with PostgreSQL',
        'version': '1.1.0',
        'status': 'running',
        'endpoints': {
            '/': 'This information',
            '/api/health': 'Health check',
            '/api/predict': 'POST - Predict transaction risk',
            '/api/generate-sar': 'POST - Generate SAR narrative',
            '/api/audit-logs': 'GET - View audit logs',
            '/api/threshold': 'GET/POST - View/update threshold',
            '/api/synthetic-examples': 'GET - View synthetic data'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    db_status = "disconnected"
    db_count = 0
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM audit_log")
        db_count = cur.fetchone()[0]
        cur.close()
        conn.close()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database': db_status,
        'audit_log_count': db_count,
        'threshold': detector.threshold
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        prediction = detector.predict(data)
        case_id = f"PRED-{datetime.now().strftime('%Y%m%d%H%M%S')}-{np.random.randint(1000, 9999)}"
        
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO audit_log 
                (case_id, input_data, model_used, feature_importance, reasoning_trace, created_by)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                case_id,
                Json(data),
                'rule-based-v1',
                Json(detector.feature_importance),
                Json({'prediction': prediction, 'rules_applied': prediction['reasons']}),
                'api_user'
            ))
            audit_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
            conn.close()
            db_message = f"Logged to audit DB (ID: {audit_id})"
        except Exception as db_error:
            db_message = f"Database logging failed: {str(db_error)}"
        
        return jsonify({
            'success': True,
            'case_id': case_id,
            'prediction': prediction,
            'input_summary': {
                'amount': data.get('amount', 'N/A'),
                'customer_id': data.get('customer_id', 'N/A'),
                'tx_count_7d': data.get('transaction_count_7d', 'N/A'),
                'unique_cp_7d': data.get('unique_counterparties_7d', 'N/A')
            },
            'audit_status': db_message
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/generate-sar', methods=['POST'])
def generate_sar():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        prediction = detector.predict(data)
        case_id = f"SAR-{datetime.now().strftime('%Y%m%d')}-{np.random.randint(1000, 9999)}"
        
        risk_level = "HIGH" if prediction['is_suspicious'] else "LOW"
        current_date = datetime.now().strftime('%Y-%m-%d')
        
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

Based on the analysis, this activity requires further investigation 
and potential filing of a complete SAR with appropriate authorities.

---
DRAFT - Requires Analyst Review and Approval
"""
        
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO audit_log 
                (case_id, input_data, model_used, feature_importance, llm_prompt, llm_response, reasoning_trace, created_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (
                case_id,
                Json(data),
                'rule-based-v1',
                Json(detector.feature_importance),
                'Generate SAR for suspicious transaction',
                narrative,
                Json({'prediction': prediction, 'rules_applied': prediction['reasons']}),
                'api_user'
            ))
            audit_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
            conn.close()
            db_message = f"Logged to audit DB (ID: {audit_id})"
        except Exception as db_error:
            db_message = f"Database logging failed: {str(db_error)}"
        
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
            'requires_review': prediction['is_suspicious'],
            'audit_status': db_message
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/audit-logs', methods=['GET'])
def get_audit_logs():
    try:
        limit = request.args.get('limit', 10, type=int)
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT id, case_id, model_used, created_at, 
                   input_data->>'amount' as amount,
                   input_data->>'customer_id' as customer
            FROM audit_log 
            ORDER BY created_at DESC 
            LIMIT %s
        """, (limit,))
        logs = cur.fetchall()
        cur.close()
        conn.close()
        
        return jsonify({'success': True, 'count': len(logs), 'logs': logs})
    
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/threshold', methods=['GET', 'POST'])
def handle_threshold():
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

@app.route('/api/synthetic-examples', methods=['GET'])
def get_synthetic_examples():
    try:
        latest_path = config.SYNTHETIC_DIR / 'fraud_samples_latest.csv'
        
        if latest_path.exists():
            df = pd.read_csv(latest_path)
            examples = df.head(10).to_dict('records')
            
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

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 BARKLEY SAR API WITH POSTGRESQL")
    print("=" * 70)
    print(f"\n📍 API: http://127.0.0.1:5000")
    print(f"📍 Health: http://127.0.0.1:5000/api/health")
    print(f"📍 Audit: http://127.0.0.1:5000/api/audit-logs")
    print(f"📍 Threshold: http://127.0.0.1:5000/api/threshold")
    print(f"📍 Synthetic: http://127.0.0.1:5000/api/synthetic-examples")
    print(f"📍 Generate SAR: http://127.0.0.1:5000/api/generate-sar")
    print("\n⏸️  Press Ctrl+C to stop")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)