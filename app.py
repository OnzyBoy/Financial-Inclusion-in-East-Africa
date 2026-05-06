import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(page_title="East Africa Financial Inclusion Predictor", layout="wide")

# Custom CSS for UI styling based on requested color scheme
# Heading color: #00668C
# Background color: #FFFEFB
# Primary Accent Color: #71C4EF
st.markdown(
    """
    <style>
    /* Background color */
    .stApp {
        background-color: #FFFEFB;
    }
    
    /* Heading colors */
    h1, h2, h3, h4, h5, h6 {
        color: #00668C !important;
    }
    
    /* Primary Accent Color for buttons */
    .stButton>button {
        background-color: #71C4EF;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #00668C;
        color: white;
    }
    
    /* Success message styling */
    .success-box {
        background-color: #e0f7fa;
        border-left: 5px solid #00668C;
        padding: 20px;
        color: #00668C;
        border-radius: 5px;
        font-weight: bold;
        font-size: 1.2em;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    
    /* Unbanked message styling */
    .predict-box {
        background-color: #ffebee;
        border-left: 5px solid #c62828;
        padding: 20px;
        color: #c62828;
        border-radius: 5px;
        font-weight: bold;
        font-size: 1.2em;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Left Sidebar
with st.sidebar:
    st.header("About the Project")
    st.write("""
    This project aims to predict the likelihood of financial inclusion in East Africa based on demographic and socio-economic factors.
    
    **Survey Context:** Based on the 2018 FinScope survey data.
    
    **Respondents:** 23,524 individuals.
    
    By identifying key drivers of financial inclusion, stakeholders can better target resources to bridge the digital divide and provide more accessible formal banking services.
    """)

# Main Body
st.title("East Africa Financial Inclusion Predictor")

# Hero Section
st.markdown("### How the model works")
st.markdown("<p style='color: #00668C; font-size: 1.1em;'>This predictive model evaluates an individual's demographic and socioeconomic profile to determine their likelihood of having a formal bank account. Simply input the details below and hit <b>Predict</b> to see the results.</p>", unsafe_allow_html=True)

st.markdown("---")

# User Input Section
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics")
    st.markdown("#### Age")
    age = st.number_input("Enter Age", min_value=18, max_value=100, value=30, step=1, label_visibility="collapsed")
    
    st.markdown("#### Gender")
    gender = st.selectbox("Select Gender", options=["Female", "Male"], label_visibility="collapsed")

with col2:
    st.subheader("Socio-Economics")
    st.markdown("#### Education Level")
    education = st.selectbox("Select Education", 
                             options=[
                                 "No formal education / Other", 
                                 "Primary education", 
                                 "Secondary education", 
                                 "Vocational/Specialised training", 
                                 "Tertiary education"
                             ], label_visibility="collapsed")
    
    st.markdown("#### Cellphone Access")
    cellphone = st.selectbox("Select Cellphone Access", options=["No", "Yes"], label_visibility="collapsed")

st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("inclusion_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model. Ensure 'inclusion_model.pkl' is in the current directory and compatible. Error: {e}")
    model = None

# Prediction Logic
if st.button("Predict"):
    if model is not None:
        # Preprocess the user inputs
        # Gender mapping: Female -> 0, Male -> 1
        gender_val = 1 if gender == "Male" else 0
        
        # Cellphone mapping: No -> 0, Yes -> 1
        cell_val = 1 if cellphone == "Yes" else 0
        
        # Education mapping based on original dataset encoding
        edu_map = {
            "No formal education / Other": 0,
            "Primary education": 1,
            "Secondary education": 2,
            "Vocational/Specialised training": 3,
            "Tertiary education": 4
        }
        edu_val = edu_map[education]
        
        # Create a 31-feature array to match the model's expected input shape
        # The model was trained on 31 scaled features, so 0 is the mean/default for missing data
        input_data = np.zeros((1, 31))
        
        # Approximate scaling based on original dataset characteristics (mean ~38.8, std ~16.5)
        scaled_age = (age - 38.8) / 16.5 
        
        # Insert the known features at their approximate original index positions
        # location_type=0, cellphone_access=1, household_size=2, age=3, gender=4, education=5
        input_data[0, 1] = cell_val
        input_data[0, 3] = scaled_age
        input_data[0, 4] = gender_val
        input_data[0, 5] = edu_val
        
        try:
            # Predict
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
            
            st.markdown("### Results")
            
            # The Visual Feedback
            if prediction == 1:
                # Success message in --accent-200 blue (#00668C)
                st.markdown('<div class="success-box">Success! The model predicts this individual is <strong>Banked</strong>.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="predict-box">The model predicts this individual is <strong>Unbanked</strong>.</div>', unsafe_allow_html=True)
                
            # Chart showing prediction probability
            st.write("**Prediction Probability**")
            
            prob_df = pd.DataFrame({
                "Status": ["Unbanked", "Banked"],
                "Probability": [probability[0], probability[1]]
            }).set_index("Status")
            
            # Small chart using st.bar_chart
            st.bar_chart(prob_df, color="#71C4EF")
            
        except ValueError as ve:
            # This catches feature mismatch errors (e.g. if the model was trained on 32 features)
            st.error(f"Prediction Error: {ve}")
            st.info("Note: The loaded model expects a different number of features. If the model was trained on the full 32-feature dataset, you will need to provide all 32 inputs or re-train the model specifically on the 4 features used in this app.")
