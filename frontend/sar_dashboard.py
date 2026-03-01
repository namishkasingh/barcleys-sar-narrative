# sar_dashboard.py - Main Streamlit Dashboard for SAR Generator

import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Barkley SAR Generator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_URL = "http://127.0.0.1:5000"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .risk-high {
        color: #DC2626;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-low {
        color: #059669;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .risk-moderate {
        color: #D97706;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .info-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🛡️ Barkley SAR Narrative Generator</h1>', unsafe_allow_html=True)
st.markdown("---")

# Initialize session state
if 'generated_sar' not in st.session_state:
    st.session_state.generated_sar = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'audit_logs' not in st.session_state:
    st.session_state.audit_logs = None

# Sidebar - Configuration and Info
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/security-checked--v1.png", width=80)
    st.title("⚙️ Controls")
    
    # API Status
    st.subheader("🔌 API Status")
    try:
        response = requests.get(f"{API_URL}/api/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            st.success(f"✅ Connected\nThreshold: {data.get('threshold', 0.7)}")
            st.caption(f"Audit Logs: {data.get('audit_log_count', 0)}")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ Cannot connect to API")
        st.info("Make sure Flask API is running on port 5000")
    
    st.markdown("---")
    
    # Threshold Control
    st.subheader("🎚️ Detection Threshold")
    try:
        response = requests.get(f"{API_URL}/api/threshold")
        if response.status_code == 200:
            current = response.json().get('current_threshold', 0.7)
            new_threshold = st.slider(
                "Adjust sensitivity",
                min_value=0.1,
                max_value=0.95,
                value=current,
                step=0.05,
                help="Lower = More alerts (higher false positives), Higher = Fewer alerts (might miss fraud)"
            )
            if new_threshold != current:
                if st.button("Update Threshold"):
                    requests.post(f"{API_URL}/api/threshold", json={"threshold": new_threshold})
                    st.success(f"Threshold updated to {new_threshold}")
                    st.rerun()
    except:
        st.warning("Threshold control unavailable")
    
    st.markdown("---")
    st.subheader("📊 Quick Stats")
    
    # Show synthetic data stats
    try:
        response = requests.get(f"{API_URL}/api/synthetic-examples")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                stats = data.get('statistics', {})
                st.metric("Total Synthetic Samples", stats.get('total_samples', 0))
                st.metric("Avg Amount", f"₹{stats.get('avg_amount', 0):,.0f}")
    except:
        pass

# Main content area - Two columns
col1, col2 = st.columns([1, 1], gap="large")

# LEFT COLUMN - Input Form
with col1:
    st.markdown("### 📝 Case Input")
    
    with st.container():
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        
        # Input form
        with st.form("sar_input_form"):
            # Customer Information
            customer_id = st.text_input("Customer ID", value="CUST-" + datetime.now().strftime("%Y%m%d"))
            
            # Transaction Details
            st.markdown("**Transaction Details**")
            
            col_a, col_b = st.columns(2)
            with col_a:
                amount = st.number_input("Amount (₹)", min_value=0, value=500000, step=10000)
                tx_count = st.number_input("Transactions (7 days)", min_value=1, value=47, step=1)
            
            with col_b:
                unique_cp = st.number_input("Unique Counterparties", min_value=1, value=47, step=1)
                is_international = st.checkbox("International Transaction")
            
            # Alert Reason
            alert_reason = st.selectbox(
                "Alert Reason",
                ["Velocity Pattern", "High Amount", "BEC Pattern", "Mobile Trojan", "DDoS Distraction"]
            )
            
            # Additional Info
            with st.expander("Additional Information (Optional)"):
                customer_tenure = st.number_input("Customer Tenure (years)", min_value=0, value=3)
                account_type = st.selectbox("Account Type", ["Savings", "Current", "Business", "Corporate"])
            
            # Submit button
            submitted = st.form_submit_button("🚀 Generate SAR Narrative", use_container_width=True)
            
            if submitted:
                # Prepare data for API
                payload = {
                    "amount": amount,
                    "customer_id": customer_id,
                    "transaction_count_7d": tx_count,
                    "unique_counterparties_7d": unique_cp,
                    "is_international": is_international,
                    "alert_reason": alert_reason,
                    "customer_tenure": customer_tenure,
                    "account_type": account_type
                }
                
                with st.spinner("Generating SAR narrative..."):
                    try:
                        # Call generate-sar endpoint
                        response = requests.post(
                            f"{API_URL}/api/generate-sar",
                            json=payload,
                            timeout=10
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.generated_sar = result
                            st.session_state.prediction_result = result.get('prediction', {})
                            st.success("✅ SAR Narrative Generated!")
                        else:
                            st.error(f"API Error: {response.status_code}")
                    except Exception as e:
                        st.error(f"Connection Error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick Test Cases
    with st.expander("🔍 Quick Test Cases"):
        st.markdown("**Click to auto-fill test data:**")
        col_t1, col_t2, col_t3 = st.columns(3)
        
        with col_t1:
            if st.button("💰 High Amount", use_container_width=True):
                st.session_state.test_case = "high"
                st.rerun()
        
        with col_t2:
            if st.button("⚡ Velocity Pattern", use_container_width=True):
                st.session_state.test_case = "velocity"
                st.rerun()
        
        with col_t3:
            if st.button("🔄 Normal", use_container_width=True):
                st.session_state.test_case = "normal"
                st.rerun()
    
    # Auto-fill logic for test cases
    if 'test_case' in st.session_state:
        if st.session_state.test_case == "high":
            st.info("Test case loaded: High Amount Transaction")
            # Note: This would need JavaScript to actually fill the form
            # For now, just showing info

# RIGHT COLUMN - Generated SAR
with col2:
    st.markdown("### 📄 Generated SAR Narrative")
    
    if st.session_state.generated_sar:
        result = st.session_state.generated_sar
        
        # Risk Level Badge
        risk_level = result.get('risk_level', 'LOW')
        risk_score = result.get('risk_score', 0)
        
        if risk_level == "HIGH":
            st.markdown(f'<p class="risk-high">🔴 HIGH RISK (Score: {risk_score})</p>', unsafe_allow_html=True)
        elif risk_level == "MODERATE":
            st.markdown(f'<p class="risk-moderate">🟡 MODERATE RISK (Score: {risk_score})</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="risk-low">🟢 LOW RISK (Score: {risk_score})</p>', unsafe_allow_html=True)
        
        # Case ID
        st.caption(f"Case ID: {result.get('case_id', 'N/A')}")
        
        # SAR Narrative (Editable)
        narrative = st.text_area(
            "Edit Narrative if needed:",
            value=result.get('narrative', ''),
            height=400,
            key="narrative_editor"
        )
        
        # Approval Buttons
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("✅ Approve SAR", use_container_width=True):
                st.success("SAR Approved! (Saved to audit trail)")
                # Here you would update the audit log with approval status
        
        with col_b:
            if st.button("🔄 Request Revisions", use_container_width=True):
                st.info("Revision requested - Analyst will review")
        
        # Audit Trail for this SAR
        with st.expander("🔍 View Audit Trail (Why this decision?)"):
            st.json(result.get('audit_trail', {}))
            
            # Feature Importance Chart
            if 'audit_trail' in result:
                audit = result['audit_trail']
                if 'feature_importance' in audit:
                    fi = audit['feature_importance']
                    if isinstance(fi, dict):
                        df_fi = pd.DataFrame({
                            'Feature': list(fi.keys()),
                            'Importance': list(fi.values())
                        })
                        fig = px.bar(df_fi, x='Importance', y='Feature', orientation='h',
                                    title='Feature Importance',
                                    color='Importance', color_continuous_scale='Blues')
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Enter case data and click 'Generate SAR Narrative' to begin")
        # Placeholder image or message
        st.image("https://img.icons8.com/color/200/000000/security-checked.png")

# AUDIT LOGS SECTION (Full width below)
st.markdown("---")
st.markdown("### 📋 Recent Audit Logs")

# Fetch and display audit logs
try:
    response = requests.get(f"{API_URL}/api/audit-logs?limit=20")
    if response.status_code == 200:
        data = response.json()
        if data.get('success') and data.get('logs'):
            logs = data['logs']
            
            # Convert to DataFrame for display
            df_logs = pd.DataFrame(logs)
            
            # Format for display
            if not df_logs.empty:
                # Select and rename columns
                display_cols = ['id', 'case_id', 'model_used', 'created_at', 'amount', 'customer']
                available_cols = [col for col in display_cols if col in df_logs.columns]
                
                st.dataframe(
                    df_logs[available_cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "id": "ID",
                        "case_id": "Case ID",
                        "model_used": "Model",
                        "created_at": "Timestamp",
                        "amount": "Amount (₹)",
                        "customer": "Customer"
                    }
                )
                
                # Summary metrics
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("Total Logs", len(logs))
                if 'amount' in df_logs.columns:
                    col_m2.metric("Avg Amount", f"₹{df_logs['amount'].astype(float).mean():,.0f}")
                    col_m3.metric("Max Amount", f"₹{df_logs['amount'].astype(float).max():,.0f}")
        else:
            st.info("No audit logs found. Make some predictions first!")
    else:
        st.warning("Could not fetch audit logs")
except Exception as e:
    st.warning(f"Audit logs unavailable: {str(e)}")

# Footer
st.markdown("---")
st.markdown("© 2026 Barkley - Prototype for Internal Demonstration")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")