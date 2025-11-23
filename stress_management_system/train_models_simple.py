"""
Simple Train All Models Script
Trains Random Forest, SVM models for stress detection (LSTM training is complex, skipping for now)
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*70)
    print("  🚀 TRAINING STRESS DETECTION MODELS (RF & SVM)")
    print("="*70)
    
    # Load preprocessed data
    data_path = "data/processed_wesad.csv"
    print(f"\n📊 Loading data from {data_path}...")
    
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found!")
        return
    
    df = pd.read_csv(data_path)
    print(f"✓ Data loaded: {len(df)} samples, {df.shape[1]} features")
    
    # Handle missing values
    print("\n🔧 Handling missing values...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    print("✓ NaN values filled")
    
    # Prepare data
    feature_cols = [col for col in df.columns if col not in ['label', 'subject']]
    X = df[feature_cols].values
    y = df['label'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📈 Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Balance with SMOTE
    print("⚖️  Balancing classes with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"✓ After SMOTE: {X_train.shape[0]} samples")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    results = {}
    
    # ===== TRAIN RANDOM FOREST =====
    print("\n" + "="*70)
    print("1️⃣  TRAINING RANDOM FOREST")
    print("="*70)
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    y_pred_rf = rf_model.predict(X_test)
    rf_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred_rf, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred_rf, average='weighted', zero_division=0)
    }
    
    print(f"✓ Accuracy:  {rf_metrics['accuracy']:.4f} ({rf_metrics['accuracy']*100:.2f}%)")
    print(f"✓ Precision: {rf_metrics['precision']:.4f}")
    print(f"✓ Recall:    {rf_metrics['recall']:.4f}")
    print(f"✓ F1-Score:  {rf_metrics['f1_score']:.4f}")
    
    results['random_forest'] = rf_metrics
    
    # Save Random Forest
    rf_path = "models/random_forest_model.pkl"
    joblib.dump(rf_model, rf_path)
    print(f"\n💾 Saved to {rf_path}")
    
    # ===== TRAIN SVM =====
    print("\n" + "="*70)
    print("2️⃣  TRAINING SVM")
    print("="*70)
    
    svm_model = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        random_state=42,
        probability=True
    )
    svm_model.fit(X_train, y_train)
    
    y_pred_svm = svm_model.predict(X_test)
    svm_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_svm),
        'precision': precision_score(y_test, y_pred_svm, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred_svm, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred_svm, average='weighted', zero_division=0)
    }
    
    print(f"✓ Accuracy:  {svm_metrics['accuracy']:.4f} ({svm_metrics['accuracy']*100:.2f}%)")
    print(f"✓ Precision: {svm_metrics['precision']:.4f}")
    print(f"✓ Recall:    {svm_metrics['recall']:.4f}")
    print(f"✓ F1-Score:  {svm_metrics['f1_score']:.4f}")
    
    results['svm'] = svm_metrics
    
    # Save SVM
    svm_path = "models/svm_model.pkl"
    joblib.dump(svm_model, svm_path)
    print(f"\n💾 Saved to {svm_path}")
    
    # ===== COMPARISON =====
    print("\n" + "="*70)
    print("📊 MODEL COMPARISON")
    print("="*70)
    
    comparison_df = pd.DataFrame(results).T
    print(comparison_df)
    
    # Save comparison
    comparison_df.to_csv("models/model_comparison.csv")
    print("\n✓ Saved to models/model_comparison.csv")
    
    # Find best model
    best_model = comparison_df['f1_score'].idxmax()
    best_f1 = comparison_df.loc[best_model, 'f1_score']
    
    print(f"\n🏆 BEST MODEL: {best_model.upper()}")
    print(f"   F1-Score: {best_f1:.4f}")
    print(f"   Accuracy: {comparison_df.loc[best_model, 'accuracy']:.4f}")
    
    # Save metadata
    import json
    metadata = {
        'models_trained': list(results.keys()),
        'best_model': best_model,
        'training_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'metrics': {
            'random_forest': {
                'accuracy': float(rf_metrics['accuracy']),
                'precision': float(rf_metrics['precision']),
                'recall': float(rf_metrics['recall']),
                'f1_score': float(rf_metrics['f1_score'])
            },
            'svm': {
                'accuracy': float(svm_metrics['accuracy']),
                'precision': float(svm_metrics['precision']),
                'recall': float(svm_metrics['recall']),
                'f1_score': float(svm_metrics['f1_score'])
            }
        }
    }
    
    with open('models/models_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✓ Metadata saved to models/models_metadata.json")
    
    print("\n" + "="*70)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*70)
    print("\nSaved models:")
    print(f"  • Random Forest: {rf_path}")
    print(f"  • SVM: {svm_path}")
    print(f"  • Scaler: models/scaler.pkl")
    print(f"  • Metadata: models/models_metadata.json")
    print(f"  • Comparison: models/model_comparison.csv")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
