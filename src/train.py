import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from preprocess import load_data, clean_data, create_preprocessor


DATA_PATH = "data/telecom_churn.csv"
MODEL_PATH = "models/churn_model.pkl"


def train_model():

    print("Loading dataset...")

    df = load_data(DATA_PATH)

    print("Cleaning dataset...")

    df = clean_data(df)

    print("Dataset shape:", df.shape)

    # Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # Create preprocessing pipeline
    preprocessor = create_preprocessor(X_train)

    # Logistic Regression
    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]
    )

    print("Training model...")

    model.fit(
        X_train,
        y_train
    )

    # Save test data for evaluation
    test_data = X_test.copy()
    test_data["Churn"] = y_test

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\nModel saved successfully:")
    print(MODEL_PATH)

    return model, X_test, y_test


if __name__ == "__main__":
    train_model()