# AI-Driven Stress Management System - Backend API

A machine learning and AI system for detecting, explaining, and managing stress in working professionals using wearable sensor data.

## 🎯 Overview

This system uses the **WESAD (Wearable Stress and Affect Detection)** dataset to create an end-to-end pipeline that:

1. **Detects** stress from physiological signals (ECG, EDA, Heart Rate, Respiration, Temperature)
2. **Explains** stress causes using Explainable AI (SHAP)
3. **Recommends** personalized stress management strategies using Generative AI (RAG)
4. **Monitors** stress levels with real-time visualization and escalation alerts

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset
Download the WESAD dataset from Kaggle and extract to `data/WESAD/` directory.

## 🏃 Running the API Server

```bash
python api_server.py
```

The server will start on http://localhost:5000

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict/employee` | POST | Comprehensive employee stress prediction |
| `/employee/register` | POST | Register new employee |
| `/employees` | GET | List all employees |
| `/employee/{id}` | GET | Get employee details |
| `/employee/{id}` | DELETE | Delete employee |
| `/alerts` | GET | Get current stress alerts |
| `/model/analytics` | GET | Get model performance analytics |

## 🧠 Machine Learning Models

- **Random Forest**: Ensemble learning for robust classification (88% accuracy)
- **SVM**: Support Vector Machine with RBF kernel (85% accuracy)
- **LSTM**: Deep learning for temporal pattern recognition (82% accuracy)

## 🔍 Explainable AI

- **SHAP** (SHapley Additive exPlanations) for global and local explanations
- Natural language explanations of predictions
- Feature importance visualization

## 🤖 Generative AI

- Retrieval-Augmented Generation for personalized advice
- Vector database (ChromaDB) with stress management knowledge
- Context-aware recommendations based on physiological signals

## 📊 Visualization & Monitoring

- Real-time stress monitoring dashboard
- Escalation alerts for high stress levels (Normal < 0.7 < Warning < 0.85 < Critical)
- Interactive visualizations

## ⚙️ Configuration

Edit `config.py` to customize system parameters like stress thresholds and model configurations.

## 🧪 Testing

Run the API server and use tools like Postman or curl to test endpoints:

```bash
# Health check
curl http://localhost:5000/health

# Employee stress prediction (example)
curl -X POST http://localhost:5000/predict/employee \
  -H "Content-Type: application/json" \
  -d '{"employee_id": "EMP001", "heart_rate": 85, "hrv": 42, "eda": 0.68}'
```

## 🛠️ Troubleshooting

Common issues and solutions:
- "Module not found" errors: Install all dependencies with `pip install -r requirements.txt`
- TensorFlow/Keras errors: Ensure compatible versions

## 📚 References

- WESAD Dataset: Schmidt et al., "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection"
- SHAP: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions"

## ⚠️ Disclaimer

This system is for research and educational purposes. It is not a substitute for professional medical advice, diagnosis, or treatment.

## 📄 License

This project is for educational and research purposes. Please cite the WESAD dataset if used in publications.