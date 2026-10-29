import joblib
import pandas as pd


MODEL_PATH = "models/churn_model.pkl"


def load_model():
    """
    Load trained churn model.
    """
    return joblib.load(MODEL_PATH)


def predict_churn(customer_data):

    model = load_model()

    # Convert dictionary to DataFrame
    customer_df = pd.DataFrame(
        [customer_data]
    )

    # Probability of churn
    probability = model.predict_proba(
        customer_df
    )[0][1]

    # Prediction
    prediction = int(
        probability >= 0.50
    )

    # Risk classification
    if probability >= 0.70:
        risk = "High Risk"

    elif probability >= 0.40:
        risk = "Medium Risk"

    else:
        risk = "Low Risk"

    return {
        "churn_prediction": prediction,
        "churn_probability": round(
            probability,
            4
        ),
        "risk_level": risk
    }


if __name__ == "__main__":

    customer = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.5,
        "TotalCharges": 286.5
    }

    result = predict_churn(customer)

    print("\nPrediction Result:")
    print(result)