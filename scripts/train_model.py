import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def train_model():
    df = pd.read_csv(r"C:\Users\manis\OneDrive\Desktop\Realtimevoice\data\final\multimodal_features.csv")

    features = [
        "sentiment", "grammar_errors", 
        "angry", "disgust", "fear", 
        "happy", "sad", "surprise", "neutral"
    ]
    target = "score"

    if target not in df.columns:
        print("⚠️ 'score' column not found. Generating dummy scores for now...")
        df['score'] = (
            0.2 * df['sentiment'] +
            0.2 * (1 - df['grammar_errors']) + 
            0.6 * (df['happy'] + df['neutral'] - df['angry'] - df['sad'])
        )
        df['score'] = (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min())

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print(" Model trained!")
    print(f"MAE: {mean_absolute_error(y_test, preds):.3f}")
    print(f"R²: {r2_score(y_test, preds):.3f}")

    # Create models directory in the main REALTIMEVOICE folder
    models_dir = r"C:\Users\manis\OneDrive\Desktop\Realtimevoice\models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the model to the REALTIMEVOICE folder
    model_path = os.path.join(models_dir, "rf_model.pkl")
    joblib.dump(model, model_path)
    print(f"📦 Model saved to {model_path}")

if __name__ == "__main__":
    train_model()