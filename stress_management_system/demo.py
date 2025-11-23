"""
Quick Demo Script
Demonstrates the AI-Driven Stress Management System with sample data
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import StressManagementPipeline


def print_banner():
    """Print welcome banner"""
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║        AI-DRIVEN STRESS MANAGEMENT SYSTEM - QUICK DEMO            ║
    ║                                                                    ║
    ║   Detect → Explain → Recommend → Monitor Stress in Real-Time     ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)


def check_data_availability():
    """Check if processed data is available"""
    processed_data = "data/processed_wesad.csv"
    model_file = "models/random_forest_model.pkl"
    
    data_exists = os.path.exists(processed_data)
    model_exists = os.path.exists(model_file)
    
    return data_exists, model_exists, processed_data, model_file


def run_quick_demo():
    """Run quick demonstration"""
    print_banner()
    
    # Check data availability
    data_exists, model_exists, data_path, model_path = check_data_availability()
    
    if not data_exists:
        print("❌ ERROR: Processed data not found!")
        print(f"   Expected location: {data_path}")
        print("\n📋 To process data, run:")
        print("   python src/pipeline.py")
        print("\nOR download WESAD dataset and run preprocessing first.")
        return
    
    print("✅ Processed data found!")
    
    # Load data
    print("\n📊 Loading processed data...")
    df = pd.read_csv(data_path)
    feature_cols = [col for col in df.columns if col not in ['label', 'subject']]
    X = df[feature_cols].values
    y = df['label'].values
    
    print(f"   Total samples: {len(df)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Class distribution: Baseline={np.sum(y==0)}, Stress={np.sum(y==1)}, Amusement={np.sum(y==2)}")
    
    # Initialize pipeline
    print("\n🚀 Initializing pipeline...")
    pipeline = StressManagementPipeline({
        'stress_threshold': 0.7,
        'escalation_threshold': 0.85
    })
    
    pipeline.feature_names = feature_cols
    
    # Setup RAG system
    knowledge_base_path = "knowledge_base"
    if os.path.exists(knowledge_base_path):
        print("   Setting up RAG system...")
        pipeline.setup(knowledge_base_path=knowledge_base_path)
    else:
        print("   ⚠️ Knowledge base not found. Skipping RAG setup.")
        pipeline.visualizer = pipeline.visualizer or pipeline.setup()[0]
    
    # Load or train model
    if model_exists:
        print(f"\n🤖 Loading existing model from {model_path}...")
        pipeline.load_model(model_path, model_type='random_forest')
    else:
        print("\n🤖 Training new Random Forest model...")
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        pipeline.model = model
        pipeline.model_type = 'random_forest'
        
        # Save model
        import joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_path)
        print(f"   ✅ Model saved to {model_path}")
    
    # Setup explainer
    print("\n🔍 Setting up XAI explainer...")
    pipeline.setup_explainer(X[:100], model_type='random_forest')
    
    # Demo predictions
    print("\n" + "="*75)
    print("🎯 DEMO: Predicting Stress with Explanations and Recommendations")
    print("="*75)
    
    # Find examples of each class
    for class_idx, class_name in enumerate(['Baseline', 'Stress', 'Amusement']):
        indices = np.where(y == class_idx)[0]
        if len(indices) > 0:
            print(f"\n\n{'='*75}")
            print(f"Example {class_idx + 1}: {class_name.upper()} Case")
            print('='*75)
            
            sample_idx = indices[0]
            result = pipeline.predict_and_explain(X[sample_idx], verbose=True)
            
            print(f"\n✓ True Label: {class_name}")
            print(f"✓ Predicted: {result['predicted_label']}")
            print(f"✓ Confidence: {result['probabilities'][result['prediction']]:.2%}")
    
    # Quick statistics
    print("\n\n" + "="*75)
    print("📈 QUICK PERFORMANCE CHECK")
    print("="*75)
    
    # Make predictions on sample
    sample_size = min(500, len(X))
    X_sample = X[:sample_size]
    y_sample = y[:sample_size]
    
    y_pred = pipeline.model.predict(X_sample)
    y_pred_proba = pipeline.model.predict_proba(X_sample)
    
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    accuracy = accuracy_score(y_sample, y_pred)
    f1 = f1_score(y_sample, y_pred, average='weighted')
    
    print(f"\nSample Size: {sample_size}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_sample, y_pred, target_names=['Baseline', 'Stress', 'Amusement']))
    
    # Stress statistics
    stress_probs = y_pred_proba[:, 1]
    high_stress_count = np.sum(stress_probs >= 0.7)
    critical_count = np.sum(stress_probs >= 0.85)
    
    print("\nStress Level Statistics:")
    print(f"  Mean stress probability: {np.mean(stress_probs):.3f}")
    print(f"  High stress cases (≥0.7): {high_stress_count} ({high_stress_count/len(stress_probs)*100:.1f}%)")
    print(f"  Critical cases (≥0.85): {critical_count} ({critical_count/len(stress_probs)*100:.1f}%)")
    
    # Final message
    print("\n\n" + "="*75)
    print("✅ DEMO COMPLETE!")
    print("="*75)
    print("\n📝 Next Steps:")
    print("   1. Run full pipeline: python src/pipeline.py")
    print("   2. View results in: results/")
    print("   3. Customize config: config.py")
    print("   4. Read documentation: README.md")
    print("\n💡 To analyze your own data:")
    print("   - Place WESAD dataset in data/WESAD/")
    print("   - Run: python src/data_preprocessing.py")
    print("   - Then run: python src/pipeline.py")
    print("\n" + "="*75 + "\n")


if __name__ == "__main__":
    try:
        run_quick_demo()
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {str(e)}")
        print("\nPlease ensure:")
        print("  1. All dependencies are installed: pip install -r requirements.txt")
        print("  2. Data is processed: python src/data_preprocessing.py")
        print("  3. Check README.md for detailed instructions")
        import traceback
        traceback.print_exc()
