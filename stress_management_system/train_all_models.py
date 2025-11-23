"""
Train All Models Script
Trains Random Forest, SVM, and LSTM models for stress detection
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from stress_detection_models import StressDetectionModels

def main():
    print("="*70)
    print("  🚀 TRAINING ALL STRESS DETECTION MODELS")
    print("="*70)
    
    # Load preprocessed data
    data_path = "data/processed_wesad.csv"
    print(f"\n📊 Loading data from {data_path}...")
    
    if not os.path.exists(data_path):
        print(f"❌ Error: {data_path} not found!")
        print("Please run the preprocessing pipeline first.")
        return
    
    df = pd.read_csv(data_path)
    print(f"✓ Data loaded: {len(df)} samples, {df.shape[1]} features")
    
    # Handle missing values
    print("\n🔧 Handling missing values...")
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        print(f"Found {nan_count} NaN values")
        # Fill NaN with column mean for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        print("✓ NaN values filled with column means")
    else:
        print("✓ No NaN values found")
    
    # Initialize trainer
    trainer = StressDetectionModels(random_state=42)
    
    # Prepare data
    print("\n📈 Preparing training and test sets...")
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        df, test_size=0.2, balance=True
    )
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    results = {}
    
    # ===== TRAIN RANDOM FOREST =====
    print("\n" + "="*70)
    print("1️⃣  TRAINING RANDOM FOREST")
    print("="*70)
    
    rf_model = trainer.train_random_forest(X_train, y_train)
    rf_metrics = trainer.evaluate_model(rf_model, X_test, y_test, 'Random_Forest')
    results['random_forest'] = rf_metrics
    
    # Save Random Forest
    rf_path = "models/random_forest_model.pkl"
    trainer.save_model('random_forest', rf_path)
    print(f"✓ Saved to {rf_path}")
    
    # ===== TRAIN SVM =====
    print("\n" + "="*70)
    print("2️⃣  TRAINING SUPPORT VECTOR MACHINE (SVM)")
    print("="*70)
    
    svm_model = trainer.train_svm(X_train, y_train)
    svm_metrics = trainer.evaluate_model(svm_model, X_test, y_test, 'SVM')
    results['svm'] = svm_metrics
    
    # Save SVM
    svm_path = "models/svm_model.pkl"
    trainer.save_model('svm', svm_path)
    print(f"✓ Saved to {svm_path}")
    
    # ===== TRAIN LSTM =====
    print("\n" + "="*70)
    print("3️⃣  TRAINING LSTM (DEEP LEARNING)")
    print("="*70)
    
    # Split training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    lstm_model, history = trainer.train_lstm(
        X_train_split, y_train_split, X_val, y_val, 
        epochs=50, batch_size=32
    )
    lstm_metrics = trainer.evaluate_model(lstm_model, X_test, y_test, 'LSTM')
    results['lstm'] = lstm_metrics
    
    # Save LSTM
    lstm_path = "models/lstm_model.h5"
    trainer.save_model('lstm', lstm_path)
    print(f"✓ Saved to {lstm_path}")
    
    # ===== COMPARISON =====
    print("\n" + "="*70)
    print("📊 MODEL COMPARISON")
    print("="*70)
    
    comparison_df = trainer.compare_models()
    
    # Save comparison results
    comparison_df.to_csv("models/model_comparison.csv")
    print("\n✓ Comparison saved to models/model_comparison.csv")
    
    # Find best model
    best_model = comparison_df['f1_score'].idxmax()
    best_f1 = comparison_df.loc[best_model, 'f1_score']
    
    print(f"\n🏆 BEST MODEL: {best_model.upper()}")
    print(f"   F1-Score: {best_f1:.4f}")
    print(f"   Accuracy: {comparison_df.loc[best_model, 'accuracy']:.4f}")
    
    # Save metadata
    metadata = {
        'models_trained': list(results.keys()),
        'best_model': best_model,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'metrics': {
            'random_forest': {
                'accuracy': float(results['random_forest']['accuracy']),
                'precision': float(results['random_forest']['precision']),
                'recall': float(results['random_forest']['recall']),
                'f1_score': float(results['random_forest']['f1_score'])
            },
            'svm': {
                'accuracy': float(results['svm']['accuracy']),
                'precision': float(results['svm']['precision']),
                'recall': float(results['svm']['recall']),
                'f1_score': float(results['svm']['f1_score'])
            },
            'lstm': {
                'accuracy': float(results['lstm']['accuracy']),
                'precision': float(results['lstm']['precision']),
                'recall': float(results['lstm']['recall']),
                'f1_score': float(results['lstm']['f1_score'])
            }
        }
    }
    
    import json
    with open('models/models_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✓ Metadata saved to models/models_metadata.json")
    
    print("\n" + "="*70)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*70)
    print("\nSaved models:")
    print(f"  • Random Forest: models/random_forest_model.pkl")
    print(f"  • SVM: models/svm_model.pkl")
    print(f"  • LSTM: models/lstm_model.h5")
    print(f"  • Scaler: models/scaler.pkl")
    print(f"  • Metadata: models/models_metadata.json")
    print(f"  • Comparison: models/model_comparison.csv")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
