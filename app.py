"""
AADB Risk Prediction System
Comprehensive DUA + IRB delay prediction using joined datasets
Random Forest ensemble model with 18 features (CV F1: 0.905)
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="AADB Risk Prediction System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .risk-critical {
        background-color: #ffebee;
        border-left-color: #d32f2f;
    }
    .risk-high {
        background-color: #fff3e0;
        border-left-color: #f57c00;
    }
    .risk-medium {
        background-color: #fff9c4;
        border-left-color: #fbc02d;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left-color: #388e3c;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path='models/risk_predictor.pkl'):
    """Load the trained model"""
    try:
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        return model_package
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please train the model first.")
        return None


def predict_risk(model_package, input_data):
    """Make risk prediction"""
    model = model_package['model']
    scaler = model_package['scaler']
    feature_columns = model_package['feature_columns']
    label_encoders = model_package['label_encoders']
    benchmarks = model_package['benchmarks']
    metadata = model_package['metadata']
    
    # Prepare features
    features_dict = {}
    
    for feature in feature_columns:
        if feature.endswith('_encoded'):
            # Categorical feature
            col = feature.replace('_encoded', '')
            if col in input_data:
                val = input_data[col]
                le = label_encoders[col]
                
                # Handle unseen categories
                if val not in le.classes_:
                    val = 'Unknown'
                if val in le.classes_:
                    encoded_val = le.transform([val])[0]
                else:
                    encoded_val = 0
                
                features_dict[feature] = encoded_val
            else:
                features_dict[feature] = 0
        
        # DUA-specific derived features
        elif feature == 'intake_ratio':
            features_dict[feature] = input_data.get('OCGM intake turnaround', 7) / 7
        elif feature == 'intake_delayed_flag':
            features_dict[feature] = 1 if input_data.get('OCGM intake turnaround', 7) > 10 else 0
        elif feature == 'pe_ratio':
            features_dict[feature] = input_data.get('Recipient PE turnaround', 30) / 30
        elif feature == 'pe_delayed_flag':
            features_dict[feature] = 1 if input_data.get('Recipient PE turnaround', 30) > 45 else 0
        elif feature == 'fe_ratio':
            features_dict[feature] = input_data.get('OCGM PE to FE turnaround', 20) / 20
        elif feature == 'total_dua_ratio':
            features_dict[feature] = input_data.get('Intake to FE (days)', 60) / 60
        elif feature == 'dua_critical_delay':
            features_dict[feature] = 1 if input_data.get('Intake to FE (days)', 60) > 90 else 0
        
        # Master tracker derived features (legacy support)
        elif feature == 'noa_consult_ratio':
            features_dict[feature] = input_data.get('days_noa_to_consult', 30) / benchmarks['noa_to_consult']
        elif feature == 'noa_consult_delayed':
            features_dict[feature] = 1 if input_data.get('days_noa_to_consult', 30) > 45 else 0
        elif feature == 'intake_submit_ratio':
            features_dict[feature] = input_data.get('days_intake_to_submit', 14) / 14
        elif feature == 'intake_submit_delayed':
            features_dict[feature] = 1 if input_data.get('days_intake_to_submit', 14) > 21 else 0
        elif feature == 'total_early_days':
            features_dict[feature] = input_data.get('days_noa_to_consult', 30) + input_data.get('days_intake_to_submit', 14)
        elif feature == 'multiple_early_delays':
            features_dict[feature] = 1 if (input_data.get('days_noa_to_consult', 30) > 45 and input_data.get('days_intake_to_submit', 14) > 21) else 0
        
        # Direct features from input
        elif feature in input_data:
            features_dict[feature] = input_data[feature]
        else:
            # Default to 0 for missing features
            features_dict[feature] = 0
    
    # Create feature array
    X_new = np.array([features_dict[col] for col in feature_columns]).reshape(1, -1)
    
    # Scale if using Logistic Regression
    if metadata['name'] == 'Logistic Regression' and scaler is not None:
        X_new = scaler.transform(X_new)
    
    # Predict
    prediction = model.predict(X_new)[0]
    probability = model.predict_proba(X_new)[0, 1]
    
    # Determine risk level
    if probability >= 0.75:
        risk_level = "Critical"
        risk_class = "risk-critical"
    elif probability >= 0.5:
        risk_level = "High"
        risk_class = "risk-high"
    elif probability >= 0.25:
        risk_level = "Medium"
        risk_class = "risk-medium"
    else:
        risk_level = "Low"
        risk_class = "risk-low"
    
    return {
        'is_high_risk': bool(prediction),
        'risk_probability': float(probability),
        'risk_level': risk_level,
        'risk_class': risk_class,
        'confidence': float(max(probability, 1 - probability)),
        'features_used': features_dict
    }


def get_recommendation(risk_level):
    """Get recommendation based on risk level"""
    recommendations = {
        'Critical': '🚨 URGENT: Immediate escalation and resource allocation required',
        'High': '⚠️ Schedule intervention meeting with stakeholders within 48 hours',
        'Medium': '📋 Monitor closely and check in with team weekly',
        'Low': '✅ Standard monitoring procedures'
    }
    return recommendations.get(risk_level, 'Continue monitoring')


def main():
    # Header
    st.markdown('<div class="main-header">📊 AADB DUA Delay Prediction System</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **Comprehensive DUA + IRB Delay Prediction System**
    
    This system predicts the risk of delays in **data access processes** for the 
    **AIM-AHEAD Data Bridge** program by analyzing both **DUA (Data Use Agreement)** and 
    **IRB (Institutional Review Board)** data using an optimized Random Forest ensemble model.
    
    **Model Performance:**
    - 📊 **Algorithm**: Random Forest (50 trees, optimized via GridSearchCV)
    - ✅ **Cross-Validation F1**: 0.905 ± 0.086 (90.5% accuracy)
    - 📈 **Training Data**: 56 samples from DUA + IRB joined datasets
      - DUA Tracker: 54 samples (100% coverage)
      - IRB Tracker: 38 samples (68% coverage, imputed where missing)
    - 🎯 **Features Used**: 18 comprehensive features (15 actively contributing):
      - **DUA Features (11)**: Timing, ratios, and delay flags
      - **IRB Features (7)**: Process status, timing, and determination flags
    
    **Top 5 Predictive Features:**
    - Total DUA ratio (25% importance)
    - PE ratio (14% importance)
    - Intake ratio (13% importance)
    - Intake delayed flag (12% importance)
    - Intake month (12% importance)
    
    **How It Works:** The ensemble model combines predictions from 50 decision trees, analyzing 
    both DUA and IRB process indicators. This comprehensive approach captures delays across the 
    entire data access workflow while ensemble averaging ensures reliable predictions.
    """)
    
    # Load model
    model_package = load_model()
    
    if model_package is None:
        st.stop()
    
    # Display model info
    with st.expander("ℹ️ Model Information", expanded=False):
        metadata = model_package['metadata']
        
        # Show data source
        if 'data_source' in metadata:
            st.info(f"**Data Source**: {metadata['data_source']}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model Type", metadata['name'])
        with col2:
            st.metric("Training Date", metadata['training_date'].strftime('%Y-%m-%d'))
        with col3:
            st.metric("Samples Trained", metadata['samples_trained'])
        with col4:
            if 'cv_f1_mean' in metadata['performance']:
                st.metric("F1 Score (CV)", f"{metadata['performance']['cv_f1_mean']:.3f}")
            else:
                st.metric("F1 Score", f"{metadata['performance']['f1']:.3f}")
    
    st.markdown("---")
    
    # Sidebar for input
    st.sidebar.header("📝 Data Access Request Details")
    st.sidebar.markdown("""
    **Enter DUA + IRB process metrics:**
    
    The model analyzes **18 comprehensive features** from both processes:
    - **DUA metrics**: Timing, ratios, acknowledgment status
    - **IRB metrics**: Determination type, timing, delay flags
    
    Top predictors: Total DUA ratio (25%), PE ratio (14%), Intake ratio (13%)
    """)
    
    # Get available programs from label encoder
    label_encoders = model_package['label_encoders']
    if 'Program' in label_encoders:
        program_options = list(label_encoders['Program'].classes_)
    else:
        program_options = ['Research Fellowship', 'Research Grant', 'Other']
    
    # Input fields - DUA specific
    program = st.sidebar.selectbox(
        "Program Type",
        options=program_options,
        help="Select the type of program/grant"
    )
    
    ocgm_intake_turnaround = st.sidebar.number_input(
        "⏱️ OCGM Intake Turnaround (days)",
        min_value=0,
        max_value=60,
        value=7,
        step=1,
        help="Days for OCGM to process the intake request (Benchmark: 7 days). Used to calculate intake_ratio (43% importance)."
    )
    
    recipient_pe_turnaround = st.sidebar.number_input(
        "📋 Recipient PE Turnaround (days)",
        min_value=0,
        max_value=120,
        value=30,
        step=1,
        help="Days for recipient to return Preliminary Execution (Benchmark: 30 days). Contributes to overall timing analysis."
    )
    
    ocgm_pe_to_fe = st.sidebar.number_input(
        "📝 OCGM PE to FE Turnaround (days)",
        min_value=0,
        max_value=60,
        value=20,
        step=1,
        help="Days from PE to Final Execution (Benchmark: 20 days). Part of the DUA completion timeline."
    )
    
    intake_to_fe = st.sidebar.number_input(
        "⚡ Total Intake to FE (days) [CRITICAL]",
        min_value=0,
        max_value=180,
        value=60,
        step=1,
        help="Total days from intake to final execution (Benchmark: 60 days). STRONGEST PREDICTOR (57% importance)!"
    )
    
    intake_month = st.sidebar.slider(
        "Intake Month",
        min_value=1,
        max_value=12,
        value=6,
        help="Month when the DUA request was received (1=Jan, 12=Dec)"
    )
    
    has_acknowledgment = st.sidebar.checkbox(
        "Recipient Acknowledgment Received",
        value=True,
        help="Has the recipient acknowledged the DUA request?"
    )
    
    # IRB Section
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏥 IRB Process Data (Optional)")
    
    is_exempt = st.sidebar.checkbox(
        "IRB Exempt Status",
        value=False,
        help="Is this request IRB exempt?"
    )
    
    is_nhsr = st.sidebar.checkbox(
        "Not Human Subjects Research (NHSR)",
        value=False,
        help="Is this classified as Not Human Subjects Research?"
    )
    
    days_consult_to_submit = st.sidebar.number_input(
        "Days: Consult to IRB Submit",
        min_value=0,
        max_value=120,
        value=14,
        step=1,
        help="Days from initial IRB consult to submission (Typical: 14 days)"
    )
    
    days_submit_to_determination = st.sidebar.number_input(
        "Days: Submit to Determination",
        min_value=0,
        max_value=120,
        value=30,
        step=1,
        help="Days from IRB submission to determination (Benchmark: 30 days)"
    )
    
    days_total_irb = st.sidebar.number_input(
        "Total IRB Process (days)",
        min_value=0,
        max_value=180,
        value=60,
        step=1,
        help="Total days for IRB process (Benchmark: 60 days)"
    )
    
    st.sidebar.markdown("---")
    
    predict_button = st.sidebar.button("🔮 Predict Risk", type="primary", width='stretch')
    
    # Main content area
    if predict_button:
        # Prepare input data - DUA + IRB features
        input_data = {
            # DUA features
            'Program': program,
            'OCGM intake turnaround': ocgm_intake_turnaround,
            'Recipient PE turnaround': recipient_pe_turnaround,
            'OCGM PE to FE turnaround': ocgm_pe_to_fe,
            'Intake to FE (days)': intake_to_fe,
            'intake_month': intake_month,
            'has_acknowledgment': int(has_acknowledgment),
            # IRB features
            'is_exempt': int(is_exempt),
            'is_nhsr': int(is_nhsr),
            'days_consult_to_submit': days_consult_to_submit,
            'days_submit_to_determination': days_submit_to_determination,
            'days_total_irb': days_total_irb
        }
        
        # Make prediction
        with st.spinner('Analyzing risk factors...'):
            result = predict_risk(model_package, input_data)
        
        # Display results
        st.success("✅ Prediction Complete!")
        
        # Risk Level Card
        st.markdown(f"""
        <div class="metric-card {result['risk_class']}">
            <h2 style="margin: 0;">Risk Level: {result['risk_level']}</h2>
            <h3 style="margin: 0.5rem 0;">Risk Probability: {result['risk_probability']:.1%}</h3>
            <p style="margin: 0.5rem 0; font-size: 1.1rem;">
                <strong>Confidence:</strong> {result['confidence']:.1%}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Metrics row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Risk Classification",
                "High Risk" if result['is_high_risk'] else "Low Risk",
                delta="Needs Attention" if result['is_high_risk'] else "On Track",
                delta_color="inverse" if result['is_high_risk'] else "normal"
            )
        
        with col2:
            st.metric(
                "Risk Score",
                f"{result['risk_probability']:.1%}",
                delta=f"{abs(result['risk_probability'] - 0.5):.1%} from threshold"
            )
        
        with col3:
            st.metric(
                "Model Confidence",
                f"{result['confidence']:.1%}",
                delta="High" if result['confidence'] > 0.75 else "Medium"
            )
        
        st.markdown("---")
        
        # Recommendation
        st.subheader("📋 Recommended Action")
        recommendation = get_recommendation(result['risk_level'])
        st.info(recommendation)
        
        # Risk breakdown
        st.subheader("🔍 Risk Analysis Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Input Summary**")
            summary_df = pd.DataFrame({
                'Parameter': ['Program', 'OCGM Intake (days)', 'Recipient PE (days)', 
                             'PE to FE (days)', 'Total Intake to FE (days)',
                             'Intake Month', 'Has Acknowledgment'],
                'Value': [str(program), str(ocgm_intake_turnaround), str(recipient_pe_turnaround), 
                         str(ocgm_pe_to_fe), str(intake_to_fe),
                         str(intake_month), 'Yes' if has_acknowledgment else 'No']
            })
            st.dataframe(summary_df, width='stretch', hide_index=True)
        
        with col2:
            # Risk probability gauge
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Create horizontal bar
            colors = ['#388e3c', '#fbc02d', '#f57c00', '#d32f2f']
            positions = [0, 0.25, 0.5, 0.75, 1.0]
            
            for i in range(len(colors)):
                ax.barh(0, positions[i+1] - positions[i], left=positions[i], 
                       height=0.3, color=colors[i], alpha=0.7)
            
            # Add marker for current risk
            ax.plot([result['risk_probability']], [0], marker='v', 
                   markersize=20, color='black', zorder=10)
            ax.text(result['risk_probability'], -0.25, 
                   f"{result['risk_probability']:.1%}", 
                   ha='center', fontsize=12, fontweight='bold')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_xlabel('Risk Probability', fontsize=12)
            ax.set_title('Risk Level Indicator', fontsize=14, fontweight='bold')
            
            # Add labels
            label_positions = [0.125, 0.375, 0.625, 0.875]
            label_names = ['Low', 'Medium', 'High', 'Critical']
            for pos, name in zip(label_positions, label_names):
                ax.text(pos, 0, name, ha='center', va='center', 
                       fontsize=10, fontweight='bold', color='white')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Feature importance or coefficients (if available)
        if 'feature_importance' in model_package['metadata']:
            st.subheader("📊 Key Risk Factors (Feature Importance)")
            
            importance_df = model_package['metadata']['feature_importance'].head(10)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=importance_df, y='feature', x='importance', 
                       palette='viridis', ax=ax)
            ax.set_xlabel('Importance Score', fontsize=12)
            ax.set_ylabel('Feature', fontsize=12)
            ax.set_title('Top 10 Most Important Features (Decision Tree)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        
        elif 'feature_coefficients' in model_package['metadata']:
            st.subheader("📊 Key Risk Factors (Feature Coefficients)")
            
            coef_df = model_package['metadata']['feature_coefficients'].head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['red' if x < 0 else 'green' for x in coef_df['coefficient']]
            sns.barplot(data=coef_df, y='feature', x='coefficient', 
                       palette=colors, ax=ax)
            ax.set_xlabel('Coefficient (Impact on Risk)', fontsize=12)
            ax.set_ylabel('Feature', fontsize=12)
            ax.set_title('Feature Coefficients (Logistic Regression)', fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            st.caption("Green bars increase risk, red bars decrease risk. All features are used by Logistic Regression.")
        
        # Download prediction report
        st.markdown("---")
        st.subheader("💾 Export Results")
        
        report_data = {
            'Prediction Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Program': [program],
            # DUA fields
            'OCGM Intake Turnaround (days)': [ocgm_intake_turnaround],
            'Recipient PE Turnaround (days)': [recipient_pe_turnaround],
            'OCGM PE to FE (days)': [ocgm_pe_to_fe],
            'Total Intake to FE (days)': [intake_to_fe],
            'Intake Month': [intake_month],
            'Has Acknowledgment': ['Yes' if has_acknowledgment else 'No'],
            # IRB fields
            'IRB Exempt': ['Yes' if is_exempt else 'No'],
            'NHSR Status': ['Yes' if is_nhsr else 'No'],
            'Days Consult to Submit': [days_consult_to_submit],
            'Days Submit to Determination': [days_submit_to_determination],
            'Total IRB Days': [days_total_irb],
            # Prediction results
            'Risk Level': [result['risk_level']],
            'Risk Probability': [f"{result['risk_probability']:.1%}"],
            'Delayed': [result['is_high_risk']],
            'Confidence': [f"{result['confidence']:.1%}"],
            'Recommendation': [recommendation]
        }
        
        report_df = pd.DataFrame(report_data)
        csv = report_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Prediction Report (CSV)",
            data=csv,
            file_name=f"risk_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            width='stretch'
        )
    
    else:
        # Show welcome message when no prediction yet
        st.info("👈 Enter project details in the sidebar and click 'Predict Risk' to get started!")
        
        # Display DUA benchmarks
        st.subheader("📏 DUA Process Benchmarks")
        benchmarks = model_package['benchmarks']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("OCGM Intake", f"{benchmarks['intake_turnaround']} days")
        with col2:
            st.metric("PE Turnaround", f"{benchmarks['pe_turnaround']} days")
        with col3:
            st.metric("PE to FE", f"{benchmarks['pe_to_fe_turnaround']} days")
        with col4:
            st.metric("NIH Target (DUA)", f"{benchmarks['dua_days']} days")
        
        st.markdown("---")
        
        # About section
        st.subheader("ℹ️ About This System")
        st.markdown("""
        **Comprehensive Data Access Delay Prediction**
        
        This system uses machine learning to predict delays across the **complete data access workflow**, 
        analyzing both **DUA (Data Use Agreement)** and **IRB (Institutional Review Board)** processes.
        
        **Data Sources:**
        - 📊 **DUA Tracker**: 54 samples (100% coverage) - Contractual process timing
        - 🏥 **IRB Tracker**: 38 samples (68% coverage) - Ethics review process
        - 🔗 **Joined Dataset**: 56 total samples with comprehensive feature set
        
        **18 Features Analyzed:**
        
        *DUA Features (11):*
        - Program type, intake month, acknowledgment status
        - Timing: OCGM intake, PE turnaround, PE to FE
        - Derived: Intake ratio, PE ratio, FE ratio, delay flags
        
        *IRB Features (7):*
        - Exempt status, NHSR classification, delay indicators
        - Timing: Consult to submit, submit to determination, total IRB days
        
        **Model Specifications:**
        - 🎯 Algorithm: Random Forest (50 decision trees)
        - ✅ Performance: 90.5% F1 Score (cross-validated)
        - 📈 Feature Usage: 15 out of 18 features actively contributing
        
        **How to Use:**
        1. Enter DUA process details (required)
        2. Add IRB process data if available (optional - will be imputed if missing)
        3. Click "Predict Risk" for comprehensive assessment
        4. Review risk level, probability, and recommendations
        5. Download report for documentation
        
        **Risk Levels:**
        - 🟢 **Low** (0-25%): Process on track, standard monitoring
        - 🟡 **Medium** (25-50%): Watch closely, prepare interventions
        - 🟠 **High** (50-75%): Active intervention recommended
        - 🔴 **Critical** (75-100%): Urgent escalation required
        
        **Top Predictors:**
        - Total DUA ratio vs benchmark (25% importance)
        - PE turnaround ratio (14% importance)
        - Intake turnaround ratio (13% importance)
        - Early delay flags and seasonal patterns
        """)


if __name__ == "__main__":
    main()

