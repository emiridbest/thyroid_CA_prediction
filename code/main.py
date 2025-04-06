import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sk_model import train_sk_model, prepare_data
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
@st.cache_resource 
def load_model():
    model, scores = train_sk_model()
    return model, scores

model, model_scores = load_model()



def generate_patient_analysis(patient_data: pd.DataFrame, prediction:str) -> str:
    """Advise based on inputs and recurrence risk for patients using OpenAI."""
    try:

        prompt = f"""
        Analyze patient profile from records and provide a detailed analysis of the patient risk of thyroid cancer recurrence {prediction}.
        The patient has the following characteristics:
        {patient_data.to_dict(orient='records')}
        The analysis should include:
        - Potential recurrence risk
        - Recommended treatment options
        - Lifestyle changes
        - Follow-up care
        - Prognosis
        - Any other relevant information
        Please provide a detailed response.
        The analysis should be based on the latest medical guidelines and research.
        The analysis should be clear and concise, suitable for a medical professional.
        The analysis should be based on the latest medical guidelines and research.
        The analysis should be clear and concise, suitable for a medical professional.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are a professional thyroid csncer analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"AI analysis failed: {str(e)}")

def fetch_patient_data(**patient_params) -> pd.DataFrame:
    """Fetch historical patient data."""
    """Create DataFrame from patient parameters."""
    try:
        # Convert to DataFrame
        patient_data = pd.DataFrame([patient_params])
    except Exception as e:
        raise Exception(f"Data preparation failed: {str(e)}")
    return patient_data



def get_recurrence_predictions(data):
    """Get thyroid cancer recurrence risk predictions."""
    try:
        # Fetch data
        data = prepare_data(data)
        data = data.drop(["Hx Radiothreapy", "Hx Smoking"], axis=1)
        
        # Train model and make predictions
        model, scores = train_sk_model()  # Unpack all two values
        predictions = model.predict(data)
        # Convert predictions to human-readable format
        predictions = ["Low Risk", "High Risk"][predictions[0]]
        
        return predictions, scores
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None

def main():
    patient_params = {
    "Age": Age,
    "Gender": Gender,
    "Smoking": Smoking,
    "Hx Smoking": Hx_Smoking,
    "Hx Radiothreapy": Hx_Radiothreapy,
    "Thyroid Function": Thyroid_Function,
    "Physical Examination": Physical_Examination,
    "Adenopathy": Adenopathy,
    "Pathology": Pathology,
    "Focality": Focality,
    "Risk": Risk,
    "T": T,
    "N": N,
    "M": M,
    "Stage": Stage,
    "Response": Response
    } 
    data = fetch_patient_data(**patient_params)
    st.title("Thyroid Recurrence Risk Analysis")
    st.sidebar.title("Patient Input")
    # Patient input fields
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            Age = st.number_input("Age:", min_value=0, max_value=120, value=30)
            Gender = st.selectbox(
                "Select Gender:",
                options=["M", "F"], 
                index=0  # Default to "M"
            )
            Smoking = st.selectbox(
                "Smoking Status:",
                options=["No", "Yes"], 
                index=0 
            )
            Hx_Smoking = st.selectbox(
                "Hx Smoking Status:",
                options=["No", "Yes"], 
                index=0 
            )
            Hx_Radiothreapy = st.selectbox(
                "Hx Radiothreapy Status:",
                options=["No", "Yes"], 
                index=0 
            )
            Thyroid_Function = st.selectbox(
                "Thyroid Function:",
                options=["Normal", "Low", "High"], 
                index=0  # Default to "Normal"
            )
            Physical_Examination = st.selectbox(
                "Physical Examination:",
                options=["Normal", "Single nodular goiter-right", "Single nodular goiter-left", "Multinodular goiter", "Diffuse goiter"], 
                index=0  
            )
            Risk = st.selectbox(
            "Risk:",
            options=["Low", "Intermediate", "High"], 
            index=0  
            )
        with col2:
        
            Adenopathy = st.selectbox(
                "Adenopathy:",
                options=["No", "Right", "Left", "Posterior", "Bilateral", "Extensive"], 
                index=0 
            )
            Pathology = st.selectbox(
                "Pathology:",
                options=["Micropapillary", "Papillary", "Follicular", "Hurthel cell"], 
                index=0  
            )
            Focality = st.selectbox(
                "Focality:",
                options=["Uni-Focal", "Multi-Focal"], 
                index=0  
            )
            T = st.selectbox(
                "T:",
                options=["T1", "T2", "T3", "T4"], 
                index=0  
            )
            N = st.selectbox(
                "N:",
                options=["N0", "N1a", "N1b"], 
                index=0  
            )
            M = st.selectbox(
                "M:",
                options=["M0", "M1"], 
                index=0  
            )
            Stage = st.selectbox(
                "Stage:",
                options=["I", "II", "III", "IVA", "IVB"], 
                index=0  
            )
            Response = st.selectbox(
                "Response:",
                options=["Excellent", "Indeterminate", "Biochemical Incomplete", "Structural Incomplete"], 
                index=0  
            )
        data = fetch_patient_data(Age, Gender, Smoking, Hx_Smoking,
                       Hx_Radiothreapy, Thyroid_Function, Physical_Examination,
                        Adenopathy, Pathology, Focality, Risk, T, N,
                        M, Stage, Response)
        # Submit button
        if st.sidebar.button("Predict", key="Predict"):
            # Get predictions
            predictions, scores = get_recurrence_predictions(data)
            if predictions is not None:
                st.success(f"Predicted Recurrence Risk: **{predictions}**")
                
                with st.expander("Model Performance Metrics", expanded=False):
                    st.write(f"Model Accuracy: {scores['accuracy']:.2f}")
                
                # Generate AI analysis
                with st.spinner("Generating clinical analysis..."):
                    ai_analysis = generate_patient_analysis(data, predictions)
                
                st.subheader("Clinical Analysis")
                st.markdown(ai_analysis)
        else:
            st.sidebar.warning("Please fill in all fields and click Predict.")
    except ValueError as ve:
        st.error(f"Value error: {str(ve)}")
        return
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return
    # Footer
    st.sidebar.markdown(
        """
        ---
        ### About
        This app provides a detailed analysis of thyroid cancer recurrence risk based on patient data.
        """
    )


if __name__ == "__main__":
    main()
