import joblib

def save_model(model, filename="global_best_model.pkl"):
    joblib.dump(model, filename)
    print(f"✓ Model saved to {filename}")