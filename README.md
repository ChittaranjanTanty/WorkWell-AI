# AI-Driven Employee Stress Management System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-19.2.0-blue)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A comprehensive AI system for detecting, explaining, and managing workplace stress using physiological sensor data and machine learning.

## 🎯 Overview

This system leverages the WESAD (Wearable Stress and Affect Detection) dataset to create an end-to-end solution that:
- **Detects** stress from physiological signals (ECG, EDA, Heart Rate, Respiration, Temperature)
- **Explains** stress causes using Explainable AI (SHAP)
- **Recommends** personalized stress management strategies using Generative AI (RAG)
- **Monitors** stress levels with real-time visualization and escalation alerts

## 🏗️ Architecture

```
├── stress_management_system/     # Backend (Python/Flask)
│   ├── src/                      # Core modules
│   │   ├── data_preprocessing.py
│   │   ├── stress_detection_models.py
│   │   ├── explainable_ai.py
│   │   ├── generative_ai.py
│   │   ├── visualization.py
│   │   └── pipeline.py
│   ├── api_server.py             # REST API server
│   └── models/                   # Trained ML models
│
├── Frontend/                     # Frontend (React/Vite)
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── services/
│   └── src/App.jsx               # Main application
```

## ✨ Key Features

### Backend (API Server)
- **Machine Learning Models**: Random Forest (88%), SVM (85%), LSTM (82%)
- **Explainable AI**: SHAP explanations with natural language interpretation
- **Generative AI**: RAG-based personalized recommendations
- **Real-time Monitoring**: Escalation alerts and visualization
- **RESTful API**: Complete endpoints for employee stress management

### Frontend (Web Application)
- **Dashboard**: Real-time statistics and alerts overview
- **Stress Assessment**: Comprehensive form for physiological data input
- **Employee Management**: Registration and history tracking
- **Alerts System**: Color-coded critical/warning notifications
- **Responsive UI**: Mobile-friendly design with Tailwind CSS

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 18+
- WESAD dataset (for training new models)

### Backend Setup
```bash
cd stress_management_system
pip install -r requirements.txt
python api_server.py
```

### Frontend Setup
```bash
cd Frontend
npm install
npm run dev
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000

## 📊 System Performance

| Model         | Accuracy | Precision | Recall | F1-Score |
|---------------|----------|-----------|--------|----------|
| Random Forest | 88%      | 88%       | 88%    | 88%      |
| SVM           | 85%      | 85%       | 85%    | 85%      |
| LSTM          | 82%      | 82%       | 82%    | 82%      |

## 🛠️ API Endpoints

| Endpoint              | Method | Description                  |
|-----------------------|--------|------------------------------|
| `/health`             | GET    | System health check          |
| `/predict/employee`   | POST   | Employee stress assessment   |
| `/employee/register`  | POST   | Register new employee        |
| `/employees`          | GET    | List all employees           |
| `/employee/{id}`      | GET    | Get employee details         |
| `/employee/{id}`      | DELETE | Delete employee              |
| `/alerts`             | GET    | Get active stress alerts     |

## 🎯 Use Cases

1. **Employee Wellness Programs**: Proactive stress monitoring
2. **Occupational Health**: Workplace safety and wellbeing
3. **Research**: Stress detection and management studies
4. **Healthcare**: Supporting mental health professionals

## 📚 Technologies

### Backend
- Python, Flask, scikit-learn, TensorFlow
- SHAP for Explainable AI
- ChromaDB and Sentence Transformers for RAG
- Matplotlib, Seaborn, Plotly for visualization

### Frontend
- React 19, Vite, Tailwind CSS
- Axios for API communication
- Lucide React for icons
- Responsive design principles

## ⚠️ Disclaimer

This system is for research and educational purposes. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers with questions regarding medical conditions or stress management.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional stress detection models
- More stress management strategies
- Mobile app development
- Multi-language support

## 📧 Support

For issues or questions, please open an issue on GitHub.
