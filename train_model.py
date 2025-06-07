import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os


def load_data(path: str = "Sleep_Data.xlsx"):
    """Load dataset from Excel file."""
    return pd.read_excel(path)


def train_model(df: pd.DataFrame):
    """Train a RandomForestRegressor using the dataset."""
    if 'Sleep_Quality_Score' not in df.columns:
        raise ValueError("Dataset must contain 'Sleep_Quality_Score' column")
    X = df.drop(columns=['Sleep_Quality_Score'])
    y = df['Sleep_Quality_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return model, mse


def save_model(model, path: str = "models/model.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def main():
    df = load_data()
    model, mse = train_model(df)
    save_model(model)
    print(f"Model trained. MSE: {mse:.4f}")


if __name__ == "__main__":
    main()
