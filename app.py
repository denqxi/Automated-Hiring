import streamlit as st
import pandas as pd
import joblib
import pickle
import os
import sqlite3
from datetime import datetime
from PIL import Image

# Set page config for a professional corporate look
st.set_page_config(
    page_title="HireFlow AI | Enterprise Recruitment Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Database Setup
DB_FILE = "assessments.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            years_experience REAL,
            skills_match_score INTEGER,
            project_count INTEGER,
            resume_length INTEGER,
            github_activity INTEGER,
            education_level TEXT,
            model_used TEXT,
            prediction TEXT,
            probability REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_assessment(data):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO assessments (
            timestamp, years_experience, skills_match_score, project_count, 
            resume_length, github_activity, education_level, 
            model_used, prediction, probability
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data['years_experience'],
        data['skills_match_score'],
        data['project_count'],
        data['resume_length'],
        data['github_activity'],
        data['education_level'],
        data['model_used'],
        data['prediction'],
        data['probability']
    ))
    conn.commit()
    conn.close()

init_db()

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

# Sidebar UI configuration
model_choice = "Random Forest"
model_paths = {
    "Random Forest": "random_forest_pipeline.pkl"
}

model_accuracies = {
    "Random Forest": 0.8980
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
tabs = st.tabs(["New Assessment", "Assessment History"])

with tabs[0]:
    st.subheader("Candidate Assessment Parameters")
    col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### Primary Qualifications")
    years_experience = st.number_input(
        "Years of Experience", 
        min_value=0.0, max_value=50.0, value=5.0, step=0.5,
        help="The total number of years of relevant industry experience documented in the resume."
    )
    skills_match_score = st.slider(
        "Skills Match Score", 0, 100, 80,
        help="Percentage match between candidate skills and the specific job requirements."
    )
    project_count = st.number_input(
        "Project Count", 
        min_value=0, max_value=100, value=3,
        help="Total number of discrete projects or major initiatives highlighted in the portfolio."
    )

with col2:
    st.markdown("#### Holistic Metrics")
    resume_length = st.number_input(
        "Resume Length", 
        min_value=0, max_value=20000, value=1500,
        help="The raw character count of the resume document, used as a proxy for detail level."
    )
    github_activity = st.number_input(
        "GitHub Activity", 
        min_value=0, max_value=10000, value=70,
        help="Calculated activity level based on GitHub commits, PRs, and repository engagement."
    )
    education_level = st.selectbox(
        "Education Level",
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
        
        # Save to database
        save_assessment({
            "years_experience": years_experience,
            "skills_match_score": skills_match_score,
            "project_count": project_count,
            "resume_length": resume_length,
            "github_activity": github_activity,
            "education_level": education_level,
            "model_used": model_choice,
            "prediction": status_text,
            "probability": float(probability) if probability is not None else None
        })
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
                    <div style="color: #64748b; font-weight: 700; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem;">Candidate Match Probability</div>
                    <div style='font-size: 3.5rem; font-weight: 900; color: #0f172a; line-height: 1;'>{f"{probability:.1%}" if probability is not None else "N/A"}</div>
                    <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 1rem;">Likelihood that this candidate matches your enterprise hiring criteria.</div>
                    <div style="margin-top: 1.5rem;">
                        <div style="height: 10px; background-color: #f1f5f9; border-radius: 5px; overflow: hidden;">
                            <div style="width: {(probability*100) if probability is not None else 0}%; height: 100%; background-color: #0f172a;"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Added: Recent Evaluations shortcut on main page
    st.markdown("<div style='margin-top: 4rem;'></div>", unsafe_allow_html=True)
    st.subheader("Recent Evaluations")
    try:
        conn = sqlite3.connect(DB_FILE)
        df_recent = pd.read_sql_query("SELECT timestamp, education_level, prediction FROM assessments ORDER BY id DESC LIMIT 3", conn)
        conn.close()
        if not df_recent.empty:
            st.table(df_recent)
            st.caption("Go to 'Assessment History' tab for full details.")
    except:
        pass

with tabs[1]:
    st.subheader("Historical Assessment Data")
    try:
        conn = sqlite3.connect(DB_FILE)
        df_history = pd.read_sql_query("SELECT * FROM assessments ORDER BY id DESC", conn)
        conn.close()
        
        if df_history.empty:
            st.info("No assessments recorded yet. Initiate an analysis to see history.")
        else:
            # Metrics Summary
            m_col1, m_col2, m_col3 = st.columns(3)
            total_screened = len(df_history)
            shortlist_count = len(df_history[df_history['prediction'] == 'SHORTLISTED'])
            shortlist_rate = shortlist_count / total_screened if total_screened > 0 else 0
            avg_exp = df_history['years_experience'].mean()
            
            m_col1.metric("Total Candidates Screened", total_screened)
            m_col2.metric("Shortlist Rate", f"{shortlist_rate:.1%}")
            m_col3.metric("Avg. Experience", f"{avg_exp:.1f} yrs")
            
            st.markdown("---")
            
            # Drop the internal ID for display
            display_df = df_history.drop(columns=['id'])
            st.dataframe(
                display_df,
                column_config={
                    "timestamp": "Time",
                    "years_experience": "Exp (Yrs)",
                    "skills_match_score": "Skills %",
                    "project_count": "Projects",
                    "resume_length": "Length",
                    "github_activity": "GitHub",
                    "education_level": "Education",
                    "model_used": "Model",
                    "prediction": "Result",
                    "probability": st.column_config.NumberColumn("Match Prob", format="%.2f")
                },
                hide_index=True,
                use_container_width=True
            )
    except Exception as e:
        st.error(f"Error loading history: {e}")

st.sidebar.markdown("---")


st.sidebar.caption("HireFlow AI v2.0 | Infrastructure Protocol IT20")

