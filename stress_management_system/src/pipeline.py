"""
End-to-End Stress Management Pipeline
Integrates all components: Data → Detection → Explainability → Recommendation → Visualization
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(__file__))

from data_preprocessing import WESADPreprocessor
from stress_detection_models import StressDetectionModels
from explainable_ai import StressExplainer
from generative_ai import StressManagementRAG
from visualization import StressVisualization


class StressManagementPipeline:
    """
    Complete end-to-end pipeline for AI-driven stress management
    """
    
    def __init__(self, config=None):
        """
        Initialize pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.preprocessor = None
        self.model = None
        self.explainer = None
        self.rag_system = None
        self.visualizer = None
        self.feature_names = None
        
        # Default thresholds
        self.stress_threshold = self.config.get('stress_threshold', 0.7)
        self.escalation_threshold = self.config.get('escalation_threshold', 0.85)
    
    def setup(self, data_path=None, knowledge_base_path=None):
        """
        Setup pipeline components
        
        Args:
            data_path: Path to WESAD dataset
            knowledge_base_path: Path to knowledge base
        """
        print("🚀 Setting up AI-Driven Stress Management Pipeline...\n")
        
        # Initialize preprocessor
        if data_path:
            self.preprocessor = WESADPreprocessor(data_path, window_size=60, overlap=30)
            print("✓ Data preprocessor initialized")
        
        # Initialize RAG system
        if knowledge_base_path:
            self.rag_system = StressManagementRAG(knowledge_base_path)
            self.rag_system.initialize()
            print("✓ RAG system initialized")
        
        # Initialize visualizer
        self.visualizer = StressVisualization(
            stress_threshold=self.stress_threshold,
            escalation_threshold=self.escalation_threshold
        )
        print("✓ Visualization system initialized")
        
        print("\n✅ Pipeline setup complete!\n")
    
    def process_data(self, subject_ids=None, save_path=None):
        """
        Process WESAD dataset
        
        Args:
            subject_ids: List of subject IDs to process
            save_path: Path to save processed data
            
        Returns:
            Processed DataFrame
        """
        print("📊 Processing WESAD dataset...\n")
        
        if self.preprocessor is None:
            raise ValueError("Preprocessor not initialized. Call setup() first.")
        
        # Process subjects
        df = self.preprocessor.process_all_subjects(subject_ids)
        
        # Normalize features
        df = self.preprocessor.normalize_features(df)
        
        # Save if path provided
        if save_path:
            self.preprocessor.save_processed_data(df, save_path)
            scaler_path = save_path.replace('.csv', '_scaler.pkl')
            self.preprocessor.save_scaler(scaler_path)
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col not in ['label', 'subject']]
        
        print(f"\n✅ Data processing complete! Total segments: {len(df)}\n")
        
        return df
    
    def train_models(self, df, model_types=['random_forest', 'svm'], save_dir=None):
        """
        Train stress detection models
        
        Args:
            df: Processed DataFrame
            model_types: List of model types to train
            save_dir: Directory to save models
            
        Returns:
            Dictionary of trained models and results
        """
        print("🤖 Training stress detection models...\n")
        
        # Initialize trainer
        trainer = StressDetectionModels(random_state=42)
        
        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(df, test_size=0.2, balance=True)
        
        results = {}
        
        # Train models
        for model_type in model_types:
            if model_type.lower() == 'random_forest':
                print("\n" + "="*60)
                model = trainer.train_random_forest(X_train, y_train)
                metrics = trainer.evaluate_model(model, X_test, y_test, 'Random_Forest')
                results['random_forest'] = {'model': model, 'metrics': metrics}
                
                if save_dir:
                    trainer.save_model('Random_Forest', os.path.join(save_dir, 'random_forest_model.pkl'))
            
            elif model_type.lower() == 'svm':
                print("\n" + "="*60)
                model = trainer.train_svm(X_train, y_train)
                metrics = trainer.evaluate_model(model, X_test, y_test, 'SVM')
                results['svm'] = {'model': model, 'metrics': metrics}
                
                if save_dir:
                    trainer.save_model('SVM', os.path.join(save_dir, 'svm_model.pkl'))
            
            elif model_type.lower() == 'lstm':
                print("\n" + "="*60)
                from sklearn.model_selection import train_test_split
                X_train_split, X_val, y_train_split, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
                model, history = trainer.train_lstm(
                    X_train_split, y_train_split, X_val, y_val, epochs=50, batch_size=32
                )
                metrics = trainer.evaluate_model(model, X_test, y_test, 'LSTM')
                results['lstm'] = {'model': model, 'metrics': metrics, 'history': history}
                
                if save_dir:
                    trainer.save_model('LSTM', os.path.join(save_dir, 'lstm_model.h5'))
        
        # Compare models
        print("\n" + "="*60)
        trainer.compare_models()
        print("="*60)
        
        # Store best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['metrics']['f1_score'])
        self.model = results[best_model_name]['model']
        print(f"\n✅ Best model: {best_model_name.upper()} (F1-Score: {results[best_model_name]['metrics']['f1_score']:.4f})\n")
        
        return results, X_test, y_test
    
    def load_model(self, model_path, model_type='random_forest'):
        """
        Load pre-trained model
        
        Args:
            model_path: Path to model file
            model_type: Type of model
        """
        print(f"📥 Loading {model_type} model from {model_path}...")
        
        if model_type.lower() == 'lstm':
            from tensorflow import keras
            self.model = keras.models.load_model(model_path)
        else:
            self.model = joblib.load(model_path)
        
        self.model_type = model_type
        print("✅ Model loaded successfully!\n")
    
    def setup_explainer(self, X_background, model_type='random_forest'):
        """
        Setup XAI explainer
        
        Args:
            X_background: Background dataset for SHAP
            model_type: Type of model
        """
        print("🔍 Setting up explainable AI module...\n")
        
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")
        
        self.explainer = StressExplainer(self.model, model_type=model_type)
        self.explainer.set_feature_names(self.feature_names)
        self.explainer.create_shap_explainer(X_background, max_samples=100)
        
        print("✅ XAI module ready!\n")
    
    def predict_and_explain(self, X_instance, verbose=True):
        """
        Make prediction and generate explanation
        
        Args:
            X_instance: Instance to predict
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary with prediction, explanation, and recommendation
        """
        # Ensure 2D
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)
        
        # Prediction
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_instance)[0]
            y_pred = np.argmax(y_pred_proba)
        else:
            # LSTM
            X_reshaped = X_instance.reshape((X_instance.shape[0], X_instance.shape[1], 1))
            y_pred_proba = self.model.predict(X_reshaped)[0]
            y_pred = np.argmax(y_pred_proba)
        
        label_names = ['Baseline', 'Stress', 'Amusement']
        predicted_label = label_names[y_pred]
        stress_probability = y_pred_proba[1]
        
        # XAI explanation
        if self.explainer:
            top_features = self.explainer.get_top_features(X_instance, top_n=5, class_idx=1)
            explanation = self.explainer.explain_in_natural_language(X_instance, y_pred_proba)
        else:
            top_features = {}
            explanation = "Explainer not configured."
        
        # GenAI recommendation
        if self.rag_system:
            contributing_factors = list(top_features.keys())
            recommendation = self.rag_system.generate_personalized_advice(
                stress_level=stress_probability,
                contributing_factors=contributing_factors
            )
            quick_tip = self.rag_system.get_quick_tip(
                "HIGH" if stress_probability > 0.7 else "MODERATE" if stress_probability > 0.4 else "LOW"
            )
        else:
            recommendation = "RAG system not configured."
            quick_tip = ""
        
        # Check escalation
        escalation_level, escalation_message = self.visualizer.check_escalation(stress_probability)
        
        result = {
            'prediction': y_pred,
            'predicted_label': predicted_label,
            'probabilities': y_pred_proba,
            'stress_probability': stress_probability,
            'top_features': top_features,
            'explanation': explanation,
            'recommendation': recommendation,
            'quick_tip': quick_tip,
            'escalation_level': escalation_level,
            'escalation_message': escalation_message
        }
        
        # Print if verbose
        if verbose:
            self.visualizer.print_escalation_alert(
                stress_probability,
                explanation,
                recommendation + "\n\n💡 Quick Tip: " + quick_tip
            )
        
        return result
    
    def run_batch_analysis(self, X_data, y_true=None, save_dir=None):
        """
        Run batch analysis on dataset
        
        Args:
            X_data: Feature data
            y_true: True labels (optional)
            save_dir: Directory to save results
            
        Returns:
            Results dictionary
        """
        print("📈 Running batch analysis...\n")
        
        # Predictions
        if hasattr(self.model, 'predict_proba'):
            y_pred_proba = self.model.predict_proba(X_data)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            X_reshaped = X_data.reshape((X_data.shape[0], X_data.shape[1], 1))
            y_pred_proba = self.model.predict(X_reshaped)
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics if true labels provided
        if y_true is not None:
            from sklearn.metrics import accuracy_score, f1_score, classification_report
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            print("Model Performance:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=['Baseline', 'Stress', 'Amusement']))
        else:
            accuracy = None
            f1 = None
        
        # Visualizations
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            # Stress probability distribution
            self.visualizer.plot_stress_probability(
                y_pred_proba,
                y_true,
                save_path=os.path.join(save_dir, 'stress_probability.png')
            )
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.visualizer.plot_feature_importance(
                    self.feature_names,
                    self.model.feature_importances_,
                    top_n=20,
                    save_path=os.path.join(save_dir, 'feature_importance.html')
                )
            
            # Session report
            session_data = {
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'true_labels': y_true,
                'accuracy': f"{accuracy:.4f}" if accuracy else "N/A"
            }
            
            self.visualizer.create_session_report(
                session_data,
                save_path=os.path.join(save_dir, 'session_report.html')
            )
            
            # Export escalation log
            self.visualizer.export_escalation_log(
                os.path.join(save_dir, 'escalation_log.csv')
            )
        
        print("\n✅ Batch analysis complete!\n")
        
        return {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'accuracy': accuracy,
            'f1_score': f1
        }
    
    def demo_single_prediction(self, X_sample, y_sample=None):
        """
        Demonstrate single prediction with full pipeline
        
        Args:
            X_sample: Single sample or index
            y_sample: True label (optional)
        """
        print("\n" + "🎯 "  + "="*70)
        print("DEMO: Single Stress Detection with Full Pipeline")
        print("="*75 + "\n")
        
        result = self.predict_and_explain(X_sample, verbose=True)
        
        if y_sample is not None:
            label_names = ['Baseline', 'Stress', 'Amusement']
            print(f"\n📌 True Label: {label_names[y_sample]}")
            print(f"📌 Predicted Label: {result['predicted_label']}")
            print(f"📌 Correct: {'✓' if y_sample == result['prediction'] else '✗'}\n")
        
        return result


if __name__ == "__main__":
    """
    Example usage of the complete pipeline
    """
    
    # Configuration
    config = {
        'stress_threshold': 0.7,
        'escalation_threshold': 0.85
    }
    
    # Paths (update these for your system)
    DATA_PATH = "../data/WESAD"
    KNOWLEDGE_BASE_PATH = "knowledge_base"  # Relative to project root
    MODELS_DIR = "../models"
    RESULTS_DIR = "../results"
    
    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initialize pipeline
    pipeline = StressManagementPipeline(config)
    
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║     AI-DRIVEN STRESS MANAGEMENT SYSTEM FOR PROFESSIONALS      ║
    ║                                                                ║
    ║     Data → Detection → Explainability → Recommendation        ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    # Check if processed data exists
    processed_data_path = "../data/processed_wesad.csv"
    
    if os.path.exists(processed_data_path):
        print("Loading existing processed data...")
        df = pd.read_csv(processed_data_path)
        feature_cols = [col for col in df.columns if col not in ['label', 'subject']]
        pipeline.feature_names = feature_cols
        pipeline.setup(knowledge_base_path=KNOWLEDGE_BASE_PATH)
    else:
        print("Processing WESAD dataset from scratch...")
        pipeline.setup(data_path=DATA_PATH, knowledge_base_path=KNOWLEDGE_BASE_PATH)
        df = pipeline.process_data(save_path=processed_data_path)
    
    # Train models or load existing
    model_path = os.path.join(MODELS_DIR, 'random_forest_model.pkl')
    
    if os.path.exists(model_path):
        print("Loading existing model...")
        pipeline.load_model(model_path, model_type='random_forest')
        
        # Prepare test data for evaluation
        from sklearn.model_selection import train_test_split
        feature_cols = [col for col in df.columns if col not in ['label', 'subject']]
        X = df[feature_cols].values
        y = df['label'].values
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        print("Training new models...")
        results, X_test, y_test = pipeline.train_models(
            df,
            model_types=['random_forest', 'svm'],
            save_dir=MODELS_DIR
        )
    
    # Setup explainer
    feature_cols = [col for col in df.columns if col not in ['label', 'subject']]
    X = df[feature_cols].values
    pipeline.setup_explainer(X[:100], model_type='random_forest')
    
    # Run batch analysis
    pipeline.run_batch_analysis(X_test, y_test, save_dir=RESULTS_DIR)
    
    # Demo single predictions
    stress_indices = np.where(y_test == 1)[0]
    if len(stress_indices) > 0:
        # Demo stress case
        idx = stress_indices[0]
        pipeline.demo_single_prediction(X_test[idx], y_test[idx])
    
    print("\n" + "="*75)
    print("✅ Pipeline execution complete!")
    print(f"📁 Results saved to: {RESULTS_DIR}")
    print("="*75 + "\n")
