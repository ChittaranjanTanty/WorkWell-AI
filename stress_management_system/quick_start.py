"""
Quick Start - Train Models with Your Preprocessed Data
Uses the CSV files you already have in Google Drive
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

print("="*70)
print("  AI-DRIVEN STRESS MANAGEMENT SYSTEM - Quick Start")
print("="*70)
print()

# Load the combined data
print("📊 Loading processed data...")
df = pd.read_csv("data/processed_wesad.csv")

print(f"✓ Loaded {len(df)} samples from {df['subject'].nunique()} subjects")
print()

# Check labels
print("Label distribution:")
print(df['label'].value_counts().sort_index())
print()

# Prepare data
print("🔧 Preparing data...")
feature_cols = [col for col in df.columns if col not in ['label', 'subject']]
X = df[feature_cols].values
y = df['label'].values

print(f"Features: {len(feature_cols)}")
print()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print()

# Train Random Forest
print("="*70)
print("🤖 Training Random Forest Model...")
print("="*70)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train, y_train)
print("✓ Training complete!")
print()

# Evaluate
print("📈 Evaluating model...")
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print()
print("Results:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print()

print("Classification Report:")
label_names = [f'Class_{i}' for i in sorted(df['label'].unique())]
print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))

# Save model
print("💾 Saving model...")
os.makedirs("models", exist_ok=True)
joblib.dump(rf_model, "models/random_forest_model.pkl")
print("✓ Saved to: models/random_forest_model.pkl")
print()

# Save scaler placeholder (for compatibility)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)  # Fit on training data
joblib.dump(scaler, "models/scaler.pkl")
print("✓ Saved scaler to: models/scaler.pkl")
print()

print("="*70)
print("✅ SUCCESS!")
print("="*70)
print()
print("Your stress detection model is ready!")
print(f"Model accuracy: {accuracy:.2%}")
print()
print("Next steps:")
print("  - View feature importance")
print("  - Test predictions on new data")
print("  - Deploy the model")
print()
