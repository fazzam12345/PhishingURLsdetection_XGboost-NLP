import streamlit as st
import joblib
from mlops.src.feature_engineering import FeatureEngineering

model_path = "../test_technique/mlops/models/model.pkl"  
model = joblib.load(model_path)
cv_path = "../test_technique/mlops/artifacts/cv.pkl"
cv = joblib.load(cv_path)

def main():
    st.title("Phishing URL Detection")
    st.write("Enter a URL to classify whether it is phishing or legitimate.")

    url_input = st.text_input("URL")
    if st.button("Classify"):
        if url_input:
            try:
                fe = FeatureEngineering(cv=cv) 
                features_df = fe.feature_engineering_streamlit(url_input)
                prediction = model.predict(features_df)
                prediction_proba = model.predict_proba(features_df)

                st.write(f"Prediction: {'Phishing' if prediction[0] == 1 else 'Legitimate'}")
                st.write(f"Probability of being Phishing: {prediction_proba[0][1]:.2f}")
                st.write(f"Probability of being Legitimate: {prediction_proba[0][0]:.2f}")
            except FileNotFoundError as e:
                st.error(str(e)) 
        else:
            st.error("Please enter a URL.")

if __name__ == "__main__":
    main()