"""
AI-Driven Stress Management System
A comprehensive ML/AI pipeline for stress detection and management
"""

__version__ = "1.0.0"
__author__ = "Stress Management AI Team"
__description__ = "AI-Driven Stress Management for Working Professionals using WESAD Dataset"

from src.data_preprocessing import WESADPreprocessor
from src.stress_detection_models import StressDetectionModels
from src.explainable_ai import StressExplainer
from src.generative_ai import StressManagementRAG
from src.visualization import StressVisualization
from src.pipeline import StressManagementPipeline

__all__ = [
    'WESADPreprocessor',
    'StressDetectionModels',
    'StressExplainer',
    'StressManagementRAG',
    'StressVisualization',
    'StressManagementPipeline'
]
