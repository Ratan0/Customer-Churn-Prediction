# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to **churn (leave the service)** based on customer demographics, service usage, contract details, and billing information.

The project includes a complete ML pipeline and a **FastAPI REST API** for real-time churn prediction.

---

## 🚀 Project Overview

Customer churn is a major business problem for telecom and subscription-based companies. Identifying customers who are likely to leave allows companies to take preventive actions such as:

- Offering discounts
- Providing personalized plans
- Improving customer support
- Providing loyalty benefits
- Addressing service-related issues

This project uses **supervised binary classification** to predict customer churn.

### Prediction

| Value | Meaning |
|---|---|
| `0` | Customer is likely to stay |
| `1` | Customer is likely to churn |

The API also returns:

- Churn probability
- Risk level

### Risk Levels

| Probability | Risk |
|---:|---|
| `< 0.40` | Low Risk |
| `0.40 – 0.69` | Medium Risk |
| `>= 0.70` | High Risk |

---

# 🧠 Machine Learning Workflow

```text
Raw Customer Data
       ↓
Data Cleaning
       ↓
Feature Preparation
       ↓
Train / Test Split
       ↓
Preprocessing
       ↓
Feature Scaling
       ↓
One-Hot Encoding
       ↓
Logistic Regression
       ↓
Model Evaluation
       ↓
Save Model
       ↓
FastAPI
       ↓
Customer Churn Prediction
```

---

# 📊 Dataset

The dataset contains telecom customer information such as:

- Gender
- Senior citizen status
- Partner status
- Dependents
- Tenure
- Phone service
- Multiple lines
- Internet service
- Online security
- Online backup
- Device protection
- Tech support
- Streaming services
- Contract type
- Paperless billing
- Payment method
- Monthly charges
- Total charges
- Churn

### Target Variable

```text
Churn
```

Mapping:

```text
Yes → 1
No  → 0
```

> **Note:** The current included dataset is a small practice dataset intended for testing and demonstrating the complete ML/API pipeline. For production or serious model benchmarking, a substantially larger real-world dataset should be used.

---

# 🛠️ Technologies Used

### Programming Language

- Python

### Data Science

- Pandas
- NumPy
- Scikit-learn
- Joblib

### Machine Learning

- Logistic Regression
- Binary Classification
- Feature Scaling
- One-Hot Encoding
- Train/Test Split
- Stratified Sampling

### API

- FastAPI
- Uvicorn
- Pydantic

### Development

- VS Code
- Git
- GitHub

---

# 📁 Project Structure

```text
Customer Churn Prediction/
│
├── data/
│   └── telecom_churn.csv
│
├── models/
│   └── churn_model.pkl
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```bash
cd "Customer Churn Prediction"
```

---

## 2. Create Virtual Environment

```bash
python -m venv .venv
```

---

## 3. Activate Virtual Environment

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, you should see:

```text
(.venv) PS G:\projects\Customer Churn Prediction>
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

# 📦 Install Dependencies

Run:

```powershell
python -m pip install -r requirements.txt
```

The main dependencies are:

```text
pandas
numpy
scikit-learn
joblib
fastapi
uvicorn
```

---

# 🧹 Data Preprocessing

The preprocessing pipeline performs the following operations:

### Remove unnecessary columns

```python
customerID
```

is removed because it is an identifier and does not provide useful predictive information.

### Convert TotalCharges

`TotalCharges` is converted to numeric values.

Invalid values are handled using the median.

### Encode Target

```text
Yes → 1
No  → 0
```

### Feature Processing

Numerical features are standardized using:

```text
StandardScaler
```

Categorical features are encoded using:

```text
OneHotEncoder
```

The transformations are implemented using a Scikit-learn `ColumnTransformer`.

---

# 🤖 Model Training

The current model uses:

```text
Logistic Regression
```

with:

```python
class_weight="balanced"
```

This helps address class imbalance between customers who churn and customers who stay.

The model is implemented inside a Scikit-learn `Pipeline`:

```text
Preprocessor
     ↓
Logistic Regression
```

The trained pipeline is saved as:

```text
models/churn_model.pkl
```

---

# ▶️ Train the Model

From the project root:

```powershell
python src/train.py
```

Expected output:

```text
Loading dataset...
Cleaning dataset...
Dataset shape: ...
Training rows: ...
Testing rows: ...

Training model...

Model trained successfully!

Model saved at:
...\models\churn_model.pkl
```

---

# 📈 Evaluate the Model

Run:

```powershell
python src/evaluate.py
```

The evaluation includes:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Classification Report

Example:

```text
==================================================
MODEL EVALUATION
==================================================

Accuracy : 0.XXXX
Precision: 0.XXXX
Recall   : 0.XXXX
F1 Score : 0.XXXX
ROC-AUC  : 0.XXXX

Confusion Matrix:
[[... ...]
 [... ...]]
```

---

# 🔮 Test a Prediction

You can also test the prediction module directly:

```powershell
python src/predict.py
```

The prediction system returns:

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.82,
  "risk_level": "High Risk"
}
```

---

# 🌐 FastAPI Application

Start the API using:

```powershell
python -m uvicorn app:app --reload
```

The API will start at:

```text
http://127.0.0.1:8000
```

---

# ❤️ API Health Check

Open:

```text
http://127.0.0.1:8000/
```

Expected response:

```json
{
  "status": "ok",
  "message": "Customer Churn Prediction API is running"
}
```

---

# 📚 Swagger API Documentation

FastAPI automatically provides interactive API documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

You can test the `/predict` endpoint directly from Swagger UI.

---

# 🔮 Prediction API

### Endpoint

```text
POST /predict
```

### Request

```json
{
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
```

---

# 💻 cURL Example

### Windows PowerShell

Use `curl.exe`:

```powershell
curl.exe -X POST `
  "http://127.0.0.1:8000/predict" `
  -H "accept: */*" `
  -H "Content-Type: application/json" `
  -d '{
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
  }'
```

---

# 📤 Example API Response

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.82,
  "risk_level": "High Risk"
}
```

### Interpretation

```text
Churn Prediction: 1
Churn Probability: 82%
Risk Level: High Risk
```

This means the model considers this customer relatively likely to churn.

---

# 🏗️ API Architecture

```text
Client
  │
  │ POST /predict
  ↓
FastAPI
  │
  ↓
Pydantic Validation
  │
  ↓
Prediction Module
  │
  ↓
Saved ML Pipeline
  │
  ├── Preprocessing
  │      ├── Scaling
  │      └── One-Hot Encoding
  │
  ↓
Logistic Regression
  │
  ↓
Churn Probability
  │
  ↓
Risk Classification
  │
  ↓
JSON Response
```

---

# 🔐 Input Validation

FastAPI/Pydantic validates incoming customer data before sending it to the ML model.

For example:

```text
tenure → integer
SeniorCitizen → integer
MonthlyCharges → float
TotalCharges → float
```

Invalid requests are rejected by the API.

---

# 📊 Evaluation Metrics

The project uses multiple classification metrics.

### Accuracy

Measures the overall percentage of correct predictions.

### Precision

Measures how many customers predicted as churners actually churned.

### Recall

Measures how many actual churners were successfully identified.

### F1 Score

Harmonic mean of precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between churn and non-churn customers across classification thresholds.

For churn prediction, **precision, recall, F1-score, and ROC-AUC should be considered along with accuracy**, especially when the target classes are imbalanced.

---

# ⚠️ Current Limitations

1. The current dataset is small and intended for project demonstration.
2. Logistic Regression is used as the baseline model.
3. No hyperparameter optimization has been implemented yet.
4. No production database integration.
5. No authentication system for the API.
6. No frontend dashboard currently included.
7. The model has not been deployed to a cloud platform.

---

# 🚀 Future Improvements

Possible improvements include:

- Use a larger real-world telecom churn dataset.
- Compare Logistic Regression with Random Forest.
- Add XGBoost or CatBoost.
- Perform hyperparameter tuning.
- Add cross-validation.
- Handle class imbalance using appropriate techniques.
- Optimize the probability threshold.
- Add SHAP-based model explainability.
- Build a Streamlit dashboard.
- Add customer segmentation.
- Add Docker support.
- Deploy the API using Render, Railway, AWS, or another cloud platform.
- Add automated model monitoring.
- Add CI/CD using GitHub Actions.

---

# 🎯 Business Use Case

A telecom company can use this system to identify customers at high risk of leaving.

For example:

```text
Customer
   ↓
Churn Probability = 82%
   ↓
High Risk
   ↓
Retention Team
   ↓
Personalized Offer / Support
```

This allows the business to focus retention efforts on customers who are most likely to churn.

---

# 📌 Project Highlights

- End-to-end Machine Learning pipeline
- Data preprocessing
- Feature transformation
- Binary classification
- Imbalanced classification handling
- Model evaluation
- Probability-based prediction
- Risk categorization
- Model serialization using Joblib
- REST API using FastAPI
- Automatic Swagger documentation
- PowerShell/cURL API testing

---

# 👨‍💻 Author

**Ratan Kumar Soni**

Aspiring Data Analyst / Data Scientist

### Skills

- Python
- Pandas
- NumPy
- Scikit-learn
- Machine Learning
- SQL
- Excel
- FastAPI
- Git & GitHub
- AI Tools

---
# 📄 License
This project is created for educational and portfolio purposes.
