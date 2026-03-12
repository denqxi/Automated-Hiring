import streamlit as st
import pandas as pd
import joblib
import pickle
import os
from PIL import Image

# Set page config for a professional corporate look
st.set_page_config(
    page_title="HireFlow AI | Enterprise Recruitment Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional CSS styling for Enterprise Dashboard
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #f8fafc;
    }
    
    .header-container {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        padding: 1rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .logo-img {
        width: 120px;
        height: 120px;
        object-fit: contain;
        margin-bottom: 1rem;
    }
    
    .brand-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .brand-tagline {
        color: #64748b;
        font-size: 1rem;
        font-weight: 500;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3.8em;
        background-color: #0f172a;
        color: white;
        font-weight: 700;
        border: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton>button:hover {
        background-color: #1e293b;
        box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.2);
        transform: translateY(-2px);
    }
    
    .prediction-container {
        padding: 3rem;
        border-radius: 20px;
        background-color: white;
        border: 1px solid #e2e8f0;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        margin-top: 2.5rem;
        animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .status-badge {
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: -0.05em;
        margin: 0.5rem 0;
        line-height: 1;
    }
    
    .sidebar-info-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    /* Perfect blending for logos on light backgrounds */
    .blend-logo {
        mix-blend-mode: multiply;
    }
    </style>
""", unsafe_allow_html=True)

def load_model(model_path):
    try:
        return joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as f:
            return pickle.load(f)

# Sidebar UI
sidebar_logo = os.path.join("assets", "logo_sidebar.png")
if os.path.exists(sidebar_logo):
    # Using raw HTML for perfect blending
    import base64
    with open(sidebar_logo, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
        st.sidebar.markdown(
            f'<img src="data:image/png;base64,{data}" class="blend-logo" style="width: 100%; margin-bottom: 2rem;">',
            unsafe_allow_html=True
        )

st.sidebar.markdown("### System Configuration")
model_choice = st.sidebar.selectbox(
    "Active Intelligence Model",
    ["Logistic Regression", "Random Forest"],
    help="Select the AI architecture to power the screening prediction."
)

model_paths = {
    "Logistic Regression": "logistic_regression_pipeline.pkl",
    "Random Forest": "random_forest_pipeline.pkl"
}

model_accuracies = {
    "Logistic Regression": 0.9063,
    "Random Forest": 0.9010
}

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div class="sidebar-info-box">
    <small style="color: #64748b; text-transform: uppercase; font-weight: 700; font-size: 0.75rem; letter-spacing: 0.05em;">Engine Status</small><br>
    <span style="color: #0f172a; font-weight: 700; font-size: 1.1rem;">{model_choice}</span><br><br>
    <small style="color: #64748b; text-transform: uppercase; font-weight: 700; font-size: 0.75rem; letter-spacing: 0.05em;">Validation Accuracy</small><br>
    <span style="color: #0f172a; font-weight: 700; font-size: 1.1rem;">{model_accuracies[model_choice]:.2%}</span>
</div>
""", unsafe_allow_html=True)

# Main Dashboard Header
hf_icon = os.path.join("assets", "hf_icon.png")
if os.path.exists(hf_icon):
    import base64
    with open(hf_icon, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 2rem;">
                <img src="data:image/png;base64,{data}" class="blend-logo" style="width: 80px;">
                <div style="transform: translateY(10px);">
                    <h1 class='brand-title' style="margin: 0; line-height: 1;">HireFlow AI</h1>
                    <p class='brand-tagline' style="margin: 0;">Enterprise Recruitment Intelligence & Predictive Assessment Engine</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
else:
    st.title("HireFlow AI")

st.markdown("---")

# Dashboard Layout
st.subheader("Candidate Assessment Parameters")
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### Primary Qualifications")
    years_experience = st.number_input(
        "Foundational Experience (Years)", 
        min_value=0.0, max_value=50.0, value=5.0, step=0.5,
        help="The total number of years of relevant industry experience documented in the resume."
    )
    skills_match_score = st.slider(
        "Technical Skills Alignment (%)", 0, 100, 80,
        help="Percentage match between candidate skills and the specific job requirements."
    )
    project_count = st.number_input(
        "Documented Project Count", 
        min_value=0, max_value=100, value=3,
        help="Total number of discrete projects or major initiatives highlighted in the portfolio."
    )

with col2:
    st.markdown("#### Holistic Metrics")
    resume_length = st.number_input(
        "Document Volume (Characters)", 
        min_value=0, max_value=20000, value=1500,
        help="The raw character count of the resume document, used as a proxy for detail level."
    )
    github_activity = st.slider(
        "Open Source Contribution Score", 0, 100, 70,
        help="Calculated activity level based on GitHub commits, PRs, and repository engagement."
    )
    education_level = st.selectbox(
        "Highest Credential Attained",
        ["Bachelors", "Masters", "PhD", "High School"],
        help="Select the candidate's highest level of formal education."
    )

st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)

# Prediction Logic & Result Presentation
if st.button("Initiate Predictive Analysis"):
    model_path = os.path.join(os.getcwd(), "models", model_paths[model_choice])
    
    if not os.path.exists(model_path):
        st.error(f"SYSTEM ERROR: Critical model resource missing at {model_path}")
    else:
        with st.status("Performing multidimensional candidate analysis...", expanded=True) as status:
            st.write("Fetching model weights...")
            model = load_model(model_path)
            st.write("Parsing candidate parameters...")
            
            input_df = pd.DataFrame([{
                "years_experience": years_experience,
                "skills_match_score": skills_match_score,
                "project_count": project_count,
                "resume_length": resume_length,
                "github_activity": github_activity,
                "education_level": education_level
            }])
            
            st.write("Running intelligence inference...")
            prediction = model.predict(input_df)[0]
            
            probability = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)
                classes = list(model.classes_)
                if "Yes" in classes:
                    probability = proba[0][classes.index("Yes")]
                elif 1 in classes:
                    probability = proba[0][classes.index(1)]
                elif True in classes:
                    probability = proba[0][classes.index(True)]
            
            status.update(label="Analysis Complete", state="complete", expanded=False)

        # Persistent Results Section
        is_shortlisted = prediction in [1, "Yes", True]
        status_text = "SHORTLISTED" if is_shortlisted else "REJECTED"
        status_color = "#059669" if is_shortlisted else "#dc2626"
        
        st.markdown(f"""
        <div class="prediction-container">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
                <div>
                    <div style="color: #64748b; font-weight: 700; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">System Recommendation</div>
                    <div class="status-badge" style="color: {status_color};">{status_text}</div>
                    <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 1rem;">Based on multidimensional algorithmic screening of provided parameters.</div>
                </div>
                <div style="border-left: 1px solid #f1f5f9; padding-left: 2rem;">
                    <div style="color: #64748b; font-weight: 700; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">Intelligence Confidence</div>
                    <div style='font-size: 3.5rem; font-weight: 900; color: #0f172a; line-height: 1;'>{f"{probability:.1%}" if probability is not None else "N/A"}</div>
                    <div style="margin-top: 1.5rem;">
                        <div style="height: 10px; background-color: #f1f5f9; border-radius: 5px; overflow: hidden;">
                            <div style="width: {(probability*100) if probability is not None else 0}%; height: 100%; background-color: #0f172a;"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.sidebar.markdown("---")


st.sidebar.caption("HireFlow AI v2.0 | Infrastructure Protocol IT20")

