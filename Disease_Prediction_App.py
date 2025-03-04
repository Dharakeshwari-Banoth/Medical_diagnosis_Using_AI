import streamlit as st
import joblib
from streamlit_option_menu import option_menu
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Change Name & Logo
st.set_page_config(page_title="Disease Prediction", page_icon="⚕️", layout="wide")

st.markdown(
    """
    <style>
    body {
        font-family: 'Arial', sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(44, 62, 80, 0.7); /* Transparent sidebar */
        color: white;
    }
    .stButton>button {
        background-color: #ff5722;
        color: white;
        font-weight: bold;
        padding: 12px 26px;
        border-radius: 10px;
        border: none;
        transition: 0.3s;
        width: 100%;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #e64a19;
        transform: scale(1.05);
    }
    .main-header {
        font-size: 44px;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 25px;
        text-shadow: 3px 3px 6px #000000;
    }
    .prediction-container {
        background-color: rgba(255, 255, 255, 0.3);
        padding: 30px;
        border-radius: 20px;
        margin-top: 25px;
        box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
        text-align: center;
    }
    .slider-label {
        font-size: 20px;
        font-weight: bold;
        color: #FFFFFF;
    }
    .get-started {
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Adding Background Image
background_image_url = "https://img.freepik.com/premium-photo/healthcare-professional-interacts-with-futuristic-medical-ai-interface-displaying-data-diagnostics_124507-300565.jpg"

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-image: url({background_image_url});
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}
[data-testid="stAppViewContainer"]::before {{
content: "";
position: absolute;
top: 0;
left: 0;
width: 100%;
height: 100%;
background-color: rgba(0, 0, 0, 0.6);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Function to load models using joblib
@st.cache_resource
def load_models():
    return {
        'diabetes': joblib.load('Models/diabetes_model.pkl'),
        'heart_disease': joblib.load('Models/heart_disease_model.pkl'),
        'parkinsons': joblib.load('Models/parkinsons_model.pkl'),
        'lung_cancer': joblib.load('Models/lungs_disease_model.pkl'),
        'thyroid': joblib.load('Models/thyroid_model.pkl')
    }

models = load_models()

with st.sidebar:
    selected = option_menu(
        "Disease Prediction",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Prediction", "Lung Cancer Prediction", "Hypo-Thyroid Prediction"],
        icons=["activity", "heart", "person", "lungs", "clipboard2-pulse"],
        menu_icon="hospital",
        default_index=0,
    )

def display_input(label, key, type="number"):
    return st.number_input(label, key=key, step=1) if type == "number" else st.text_input(label, key=key)

def predict_disease(model_key, inputs):
    if any(i == "" or i is None for i in inputs):
        st.error("⚠️ Please fill in all fields before predicting.")
    else:
        input_array = np.array([inputs], dtype=np.float64)
        prediction = models[model_key].predict(input_array)
        diagnosis = f'The person has {selected}' if prediction[0] == 1 else f'The person does not have {selected}'
        st.success(diagnosis)

# Example: Diabetes Prediction
if selected == 'Diabetes Prediction':
    st.write("Enter the following details to predict diabetes:")
    
    inputs = [
        display_input('Number of Pregnancies', 'Pregnancies'),
        display_input('Glucose Level', 'Glucose'),
        display_input('Blood Pressure', 'BloodPressure'),
        display_input('Skin Thickness', 'SkinThickness'),
        display_input('Insulin Level', 'Insulin'),
        display_input('BMI', 'BMI'),
        display_input('Diabetes Pedigree Function', 'DiabetesPedigreeFunction'),
        display_input('Age', 'Age')
    ]
    
    if st.button('Diabetes Test Result'):
        predict_disease('diabetes', inputs)
