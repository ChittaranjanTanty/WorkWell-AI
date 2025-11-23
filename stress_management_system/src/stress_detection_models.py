"""
Stress Detection Models Module
Implements Random Forest, SVM, and LSTM models for stress classification
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class StressDetectionModels:
    """
    Stress detection model trainer and evaluator
    """
    
    def __init__(self, random_state=42):
        """Initialize models"""
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def prepare_data(self, df, test_size=0.2, balance=True):
        """
        Prepare data for training
        
        Args:
            df: Processed DataFrame
            test_size: Test set size ratio
            balance: Whether to balance classes using SMOTE
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Separate features and labels
        feature_cols = [col for col in df.columns if col not in ['label', 'subject']]
        X = df[feature_cols].values
        y = df['label'].values
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Balance classes using SMOTE
        if balance:
            print("Balancing classes using SMOTE...")
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE - Training set size: {len(X_train)}")
        
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Class distribution in training: {np.bincount(y_train)}")
        print(f"Class distribution in test: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, y_train, **kwargs):
        """
        Train Random Forest classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional parameters for RandomForestClassifier
            
        Returns:
            Trained model
        """
        print("\n=== Training Random Forest ===")
        
        default_params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        default_params.update(kwargs)
        
        rf = RandomForestClassifier(**default_params)
        rf.fit(X_train, y_train)
        
        self.models['random_forest'] = rf
        print("Random Forest training completed")
        
        return rf
    
    def train_svm(self, X_train, y_train, **kwargs):
        """
        Train SVM classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional parameters for SVC
            
        Returns:
            Trained model
        """
        print("\n=== Training SVM ===")
        
        default_params = {
            'kernel': 'rbf',
            'C': 10.0,
            'gamma': 'scale',
            'random_state': self.random_state,
            'probability': True  # Enable probability estimates for SHAP
        }
        default_params.update(kwargs)
        
        svm = SVC(**default_params)
        svm.fit(X_train, y_train)
        
        self.models['svm'] = svm
        print("SVM training completed")
        
        return svm
    
    def create_lstm_model(self, input_shape, num_classes=3, units=128, dropout=0.3):
        """
        Create LSTM model architecture
        
        Args:
            input_shape: Input shape (timesteps, features)
            num_classes: Number of output classes
            units: LSTM units
            dropout: Dropout rate
            
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Bidirectional(LSTM(units, return_sequences=True), input_shape=input_shape),
            Dropout(dropout),
            Bidirectional(LSTM(units // 2)),
            Dropout(dropout),
            Dense(64, activation='relu'),
            Dropout(dropout / 2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_lstm(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train LSTM classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Trained model and training history
        """
        print("\n=== Training LSTM ===")
        
        # Reshape for LSTM (samples, timesteps, features)
        # We'll treat each feature as a timestep
        X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val_lstm = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train)
        y_val_cat = to_categorical(y_val)
        
        num_classes = y_train_cat.shape[1]
        
        # Create model
        model = self.create_lstm_model(
            input_shape=(X_train_lstm.shape[1], 1),
            num_classes=num_classes,
            units=128,
            dropout=0.3
        )
        
        print(model.summary())
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('best_lstm_model.h5', monitor='val_accuracy', save_best_only=True)
        ]
        
        # Train
        history = model.fit(
            X_train_lstm, y_train_cat,
            validation_data=(X_val_lstm, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.models['lstm'] = model
        print("LSTM training completed")
        
        return model, history
    
    def evaluate_model(self, model, X_test, y_test, model_name='Model'):
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        print(f"\n=== Evaluating {model_name} ===")
        
        # Predictions
        if model_name.lower() == 'lstm':
            X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            y_pred_proba = model.predict(X_test_reshaped)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        target_names = ['Baseline', 'Stress', 'Amusement']
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
        
        self.results[model_name] = metrics
        
        return metrics
    
    def plot_confusion_matrix(self, cm, model_name, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Baseline', 'Stress', 'Amusement'],
                    yticklabels=['Baseline', 'Stress', 'Amusement'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, history, save_path=None):
        """Plot LSTM training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, model_name, save_path):
        """Save trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if model_name.lower() == 'lstm':
            model.save(save_path)
        else:
            joblib.dump(model, save_path)
        
        print(f"Saved {model_name} to {save_path}")
    
    def load_model(self, model_name, model_path):
        """Load trained model"""
        if model_name.lower() == 'lstm':
            model = keras.models.load_model(model_path)
        else:
            model = joblib.load(model_path)
        
        self.models[model_name] = model
        print(f"Loaded {model_name} from {model_path}")
        
        return model
    
    def compare_models(self):
        """Compare all trained models"""
        if not self.results:
            print("No models have been evaluated yet")
            return
        
        print("\n=== Model Comparison ===")
        comparison_df = pd.DataFrame(self.results).T
        print(comparison_df[['accuracy', 'precision', 'recall', 'f1_score']])
        
        return comparison_df


if __name__ == "__main__":
    # Example usage
    print("Loading processed data...")
    df = pd.read_csv("../data/processed_wesad.csv")
    
    # Initialize model trainer
    trainer = StressDetectionModels(random_state=42)
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df, test_size=0.2, balance=True)
    
    # Train Random Forest
    rf_model = trainer.train_random_forest(X_train, y_train)
    trainer.evaluate_model(rf_model, X_test, y_test, 'Random_Forest')
    
    # Train SVM
    svm_model = trainer.train_svm(X_train, y_train)
    trainer.evaluate_model(svm_model, X_test, y_test, 'SVM')
    
    # Train LSTM
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    lstm_model, history = trainer.train_lstm(X_train_split, y_train_split, X_val, y_val, epochs=50, batch_size=32)
    trainer.evaluate_model(lstm_model, X_test, y_test, 'LSTM')
    
    # Compare models
    trainer.compare_models()
    
    # Save models
    trainer.save_model('Random_Forest', '../models/random_forest_model.pkl')
    trainer.save_model('SVM', '../models/svm_model.pkl')
    trainer.save_model('LSTM', '../models/lstm_model.h5')
