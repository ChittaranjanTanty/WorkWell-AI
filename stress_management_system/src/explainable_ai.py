"""
Explainable AI (XAI) Module
Uses SHAP and LIME to explain stress detection model predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime import lime_tabular
import joblib
import warnings
warnings.filterwarnings('ignore')


class StressExplainer:
    """
    Explainable AI module for stress detection models
    """
    
    def __init__(self, model, model_type='random_forest'):
        """
        Initialize explainer
        
        Args:
            model: Trained model
            model_type: Type of model ('random_forest', 'svm', 'lstm')
        """
        self.model = model
        self.model_type = model_type.lower()
        self.explainer = None
        self.feature_names = None
        
    def set_feature_names(self, feature_names):
        """Set feature names for explanations"""
        self.feature_names = feature_names
    
    def create_shap_explainer(self, X_background, max_samples=100):
        """
        Create SHAP explainer
        
        Args:
            X_background: Background dataset for SHAP
            max_samples: Maximum samples for background
        """
        print("Creating SHAP explainer...")
        
        # Sample background data if too large
        if len(X_background) > max_samples:
            indices = np.random.choice(len(X_background), max_samples, replace=False)
            X_background = X_background[indices]
        
        if self.model_type == 'random_forest':
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == 'svm':
            self.explainer = shap.KernelExplainer(self.model.predict_proba, X_background)
        elif self.model_type == 'lstm':
            def model_predict(x):
                x_reshaped = x.reshape((x.shape[0], x.shape[1], 1))
                return self.model.predict(x_reshaped)
            self.explainer = shap.KernelExplainer(model_predict, X_background)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        print("SHAP explainer created")
    
    def explain_instance_shap(self, X_instance, instance_idx=0):
        """
        Explain a single prediction using SHAP
        
        Args:
            X_instance: Instance to explain (can be array of instances)
            instance_idx: Index of instance if multiple
            
        Returns:
            SHAP values
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer not created. Call create_shap_explainer first.")
        
        # Ensure 2D array
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X_instance)
        
        return shap_values
    
    def plot_shap_waterfall(self, X_instance, instance_idx=0, class_idx=1, save_path=None):
        """
        Plot SHAP waterfall plot for a single instance
        
        Args:
            X_instance: Instance to explain
            instance_idx: Instance index
            class_idx: Class to explain (1 = Stress)
            save_path: Path to save plot
        """
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)
        
        shap_values = self.explain_instance_shap(X_instance)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            shap_values_class = shap_values[class_idx]
        else:
            shap_values_class = shap_values
        
        # Create explanation object
        if self.feature_names is not None:
            feature_names = self.feature_names
        else:
            feature_names = [f'Feature_{i}' for i in range(X_instance.shape[1])]
        
        plt.figure(figsize=(10, 6))
        
        # Get top features
        if shap_values_class.ndim > 1:
            shap_vals = shap_values_class[instance_idx]
        else:
            shap_vals = shap_values_class
        
        # Sort by absolute value
        sorted_indices = np.argsort(np.abs(shap_vals))[::-1][:15]
        
        # Plot
        sorted_vals = shap_vals[sorted_indices]
        sorted_features = [feature_names[i] for i in sorted_indices]
        
        colors = ['red' if v > 0 else 'blue' for v in sorted_vals]
        plt.barh(range(len(sorted_vals)), sorted_vals, color=colors)
        plt.yticks(range(len(sorted_vals)), sorted_features)
        plt.xlabel('SHAP Value (impact on model output)')
        plt.title(f'Feature Importance for Stress Prediction')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
    
    def plot_shap_summary(self, X_data, max_display=20, save_path=None):
        """
        Plot SHAP summary plot
        
        Args:
            X_data: Dataset to explain
            max_display: Maximum features to display
            save_path: Path to save plot
        """
        print("Generating SHAP summary plot...")
        
        shap_values = self.explain_instance_shap(X_data)
        
        # Handle different formats
        if isinstance(shap_values, list):
            # Multi-class: use stress class (index 1)
            shap_values_plot = shap_values[1]
        else:
            shap_values_plot = shap_values
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_plot, 
            X_data, 
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_top_features(self, X_instance, top_n=5, class_idx=1):
        """
        Get top contributing features for a prediction
        
        Args:
            X_instance: Instance to explain
            top_n: Number of top features to return
            class_idx: Class index (1 = Stress)
            
        Returns:
            Dictionary of top features and their contributions
        """
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)
        
        shap_values = self.explain_instance_shap(X_instance)
        
        # Handle different formats
        if isinstance(shap_values, list):
            shap_vals = shap_values[class_idx][0]
        else:
            if shap_values.ndim > 1:
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values
        
        # Get top features
        top_indices = np.argsort(np.abs(shap_vals))[::-1][:top_n]
        
        if self.feature_names is not None:
            feature_names = self.feature_names
        else:
            feature_names = [f'Feature_{i}' for i in range(len(shap_vals))]
        
        top_features = {}
        for idx in top_indices:
            feature_name = feature_names[idx]
            contribution = shap_vals[idx]
            feature_value = X_instance[0, idx]
            
            top_features[feature_name] = {
                'contribution': float(contribution),
                'value': float(feature_value),
                'impact': 'increases' if contribution > 0 else 'decreases'
            }
        
        return top_features
    
    def explain_in_natural_language(self, X_instance, y_pred_proba):
        """
        Generate natural language explanation
        
        Args:
            X_instance: Instance to explain
            y_pred_proba: Prediction probabilities
            
        Returns:
            String explanation
        """
        # Get top features
        top_features = self.get_top_features(X_instance, top_n=5, class_idx=1)
        
        # Get prediction
        stress_prob = y_pred_proba[1] if len(y_pred_proba) > 1 else y_pred_proba[0]
        stress_level = "HIGH" if stress_prob > 0.7 else "MODERATE" if stress_prob > 0.4 else "LOW"
        
        # Build explanation
        explanation = f"Stress Level Detected: {stress_level} (probability: {stress_prob:.2%})\n\n"
        explanation += "Key Contributing Factors:\n"
        
        for i, (feature, info) in enumerate(top_features.items(), 1):
            # Simplify feature names for better readability
            simple_name = self._simplify_feature_name(feature)
            impact_word = "increasing" if info['impact'] == 'increases' else "reducing"
            
            explanation += f"{i}. {simple_name}: {info['value']:.3f} ({impact_word} stress by {abs(info['contribution']):.3f})\n"
        
        # Add interpretation
        explanation += "\nInterpretation:\n"
        
        # Check for specific physiological indicators
        hrv_features = [k for k in top_features.keys() if 'hrv' in k.lower()]
        eda_features = [k for k in top_features.keys() if 'eda' in k.lower()]
        
        if hrv_features:
            explanation += "- Heart rate variability indicators suggest "
            if top_features[hrv_features[0]]['impact'] == 'increases':
                explanation += "elevated stress response.\n"
            else:
                explanation += "normal stress levels.\n"
        
        if eda_features:
            explanation += "- Electrodermal activity (skin conductance) shows "
            if top_features[eda_features[0]]['impact'] == 'increases':
                explanation += "increased sympathetic nervous system activation.\n"
            else:
                explanation += "normal arousal levels.\n"
        
        return explanation
    
    def _simplify_feature_name(self, feature_name):
        """Simplify feature names for readability"""
        mapping = {
            'ecg': 'ECG',
            'eda': 'Skin Conductance (EDA)',
            'emg': 'Muscle Activity (EMG)',
            'temp': 'Temperature',
            'resp': 'Respiration',
            'hrv': 'Heart Rate Variability',
            'mean': 'Average',
            'std': 'Variability',
            'rms': 'RMS',
            'psd': 'Power Spectral Density',
            'hr_mean': 'Heart Rate',
            'rmssd': 'HRV RMSSD',
            'sdnn': 'HRV SDNN',
            'pnn50': 'HRV PNN50'
        }
        
        for key, value in mapping.items():
            if key in feature_name.lower():
                return feature_name.replace(key, value)
        
        return feature_name
    
    def create_lime_explainer(self, X_train, class_names=None):
        """
        Create LIME explainer
        
        Args:
            X_train: Training data
            class_names: Names of classes
        """
        if class_names is None:
            class_names = ['Baseline', 'Stress', 'Amusement']
        
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            class_names=class_names,
            mode='classification'
        )
        
        print("LIME explainer created")
    
    def explain_instance_lime(self, X_instance, num_features=10):
        """
        Explain instance using LIME
        
        Args:
            X_instance: Instance to explain
            num_features: Number of features to show
            
        Returns:
            LIME explanation
        """
        if not hasattr(self, 'lime_explainer'):
            raise ValueError("LIME explainer not created. Call create_lime_explainer first.")
        
        if X_instance.ndim > 1:
            X_instance = X_instance.flatten()
        
        # Get prediction function
        if self.model_type == 'lstm':
            def predict_fn(x):
                x_reshaped = x.reshape((x.shape[0], x.shape[1], 1))
                return self.model.predict(x_reshaped)
        else:
            predict_fn = self.model.predict_proba
        
        explanation = self.lime_explainer.explain_instance(
            X_instance,
            predict_fn,
            num_features=num_features
        )
        
        return explanation


if __name__ == "__main__":
    # Example usage
    print("Loading model and data...")
    
    # Load model
    model = joblib.load("../models/random_forest_model.pkl")
    
    # Load data
    df = pd.read_csv("../data/processed_wesad.csv")
    feature_cols = [col for col in df.columns if col not in ['label', 'subject']]
    X = df[feature_cols].values
    y = df['label'].values
    
    # Create explainer
    explainer = StressExplainer(model, model_type='random_forest')
    explainer.set_feature_names(feature_cols)
    
    # Create SHAP explainer
    explainer.create_shap_explainer(X[:100])
    
    # Explain a stress instance
    stress_indices = np.where(y == 1)[0]
    if len(stress_indices) > 0:
        sample_idx = stress_indices[0]
        X_sample = X[sample_idx]
        
        # Get prediction
        y_pred_proba = model.predict_proba(X_sample.reshape(1, -1))[0]
        
        # Generate explanation
        print("\n" + "="*50)
        explanation = explainer.explain_in_natural_language(X_sample.reshape(1, -1), y_pred_proba)
        print(explanation)
        print("="*50)
        
        # Plot SHAP
        explainer.plot_shap_waterfall(X_sample, save_path="../results/shap_waterfall.png")
