import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Tourism Package Prediction", layout="centered")

MODEL_REPO_ID = "nittygritty2106/travelpredictionmlops"
MODEL_FILENAME = "model.joblib"


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model"
    )
    return joblib.load(model_path)


model = load_model()

st.title("Tourism Package Prediction")
st.write("Predict whether a customer is likely to purchase the tourism package.")

with st.form("prediction_form"):
    Age = st.number_input("Age", min_value=18, max_value=100, value=30)
    TypeofContact = st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("CityTier", [1, 2, 3])
    DurationOfPitch = st.number_input("DurationOfPitch", min_value=1, max_value=100, value=15)
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=10, value=2)
    NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=0, max_value=10, value=3)
    ProductPitched = st.selectbox("ProductPitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    PreferredPropertyStar = st.selectbox("PreferredPropertyStar", [3, 4, 5])
    MaritalStatus = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced", "Unmarried"])
    NumberOfTrips = st.number_input("NumberOfTrips", min_value=0, max_value=20, value=2)
    Passport = st.selectbox("Passport", [0, 1])
    PitchSatisfactionScore = st.selectbox("PitchSatisfactionScore", [1, 2, 3, 4, 5])
    OwnCar = st.selectbox("OwnCar", [0, 1])
    NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=5, value=0)
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    MonthlyIncome = st.number_input("MonthlyIncome", min_value=1000, max_value=1000000, value=25000)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([{
        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch,
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfFollowups": NumberOfFollowups,
        "ProductPitched": ProductPitched,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,
        "Passport": Passport,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "OwnCar": OwnCar,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation,
        "MonthlyIncome": MonthlyIncome
    }])

    prediction = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_df)[0][1]
        st.write(f"Purchase Probability: {probability:.2f}")

    if prediction == 1:
        st.success("Customer is likely to purchase the tourism package.")
    else:
        st.error("Customer is unlikely to purchase the tourism package.")
