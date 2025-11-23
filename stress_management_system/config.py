"""
Configuration file for AI-Driven Stress Management System
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "knowledge_base")

# Data preprocessing parameters
WINDOW_SIZE = 60  # seconds
OVERLAP = 30  # seconds
SAMPLING_RATE = 700  # Hz for chest sensor
WRIST_SAMPLING_RATE = 32  # Hz for wrist sensor

# Features to extract
CHEST_FEATURES = ['ECG', 'EDA', 'EMG', 'Temp', 'Resp']
WRIST_FEATURES = ['ACC', 'BVP', 'EDA', 'TEMP']

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Stress labels mapping
LABEL_MAPPING = {
    0: 'Baseline',
    1: 'Stress',
    2: 'Amusement',
    3: 'Meditation'
}

# Use simplified mapping for binary/ternary classification
SIMPLIFIED_LABELS = {
    0: 'Baseline',
    1: 'Stress',
    2: 'Amusement'
}

# Stress threshold for escalation
STRESS_THRESHOLD = 0.7  # Probability threshold for high stress
ESCALATION_THRESHOLD = 0.85  # Critical stress level

# XAI parameters
SHAP_SAMPLES = 100
LIME_SAMPLES = 1000

# GenAI/RAG parameters
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 3

# Visualization parameters
PLOT_DPI = 300
FIGURE_SIZE = (12, 8)

# Model hyperparameters
LSTM_CONFIG = {
    'units': 128,
    'dropout': 0.3,
    'recurrent_dropout': 0.2,
    'epochs': 50,
    'batch_size': 32
}

RF_CONFIG = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE
}

SVM_CONFIG = {
    'kernel': 'rbf',
    'C': 10.0,
    'gamma': 'scale',
    'random_state': RANDOM_STATE
}
