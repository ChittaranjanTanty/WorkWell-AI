# Employee Stress Monitoring System - Frontend

A modern, responsive web application for monitoring employee stress levels using AI-powered predictions.

## 🎯 Overview

This React-based frontend provides a comprehensive user interface for the AI-Driven Stress Management System, featuring:

- Real-time stress monitoring dashboard
- Employee registration and management
- Comprehensive stress assessment forms
- Alert system with color-coded notifications
- Personalized stress management recommendations

## 🏗️ Project Structure

```
Frontend/
├── src/
│   ├── components/           # Reusable UI components
│   ├── pages/                # Application pages
│   ├── services/             # API service layer
│   ├── App.jsx               # Main app component
│   └── main.jsx              # Entry point
├── index.html
├── vite.config.js
└── package.json
```

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- Backend API running on `http://localhost:5000`

### Installation
```bash
npm install
```

### Development
```bash
npm run dev
```

The application will be available at: http://localhost:3000

### Build for Production
```bash
npm run build
npm run preview
```

## 🎨 Key Features

### Dashboard
- Real-time statistics overview
- Active alerts display
- System health status

### Stress Assessment
- Comprehensive form for physiological data input
- Work context factors
- Real-time ML predictions
- Personalized recommendations

### Employee Management
- Register new employees
- View all employees
- Search and filter capabilities
- Delete employees

### Alerts System
- Real-time critical and warning alerts
- Color-coded by severity (🟢 Normal, 🟡 Warning, 🔴 Critical)

## 🔌 API Integration

The frontend connects to the backend API through a proxy configuration to avoid CORS issues:

```javascript
// API calls are made to /api/* which proxies to http://localhost:5000/*
const response = await axios.post('/api/predict/employee', data);
```

## 🛠️ Technologies

- React 19 with Vite
- Tailwind CSS for styling
- Axios for HTTP requests
- Lucide React for icons

## 🧪 Testing the Application

1. Start the backend API server first
2. Run the frontend development server
3. Navigate to http://localhost:3000
4. Register an employee and perform a stress assessment

Example stress assessment data:
- Heart Rate: 85
- HRV: 42
- EDA: 0.68
- Temperature: 36.9
- Respiration: 19
- Activity: 0.45

## 🐛 Troubleshooting

- **CORS Errors**: Ensure you're using the Vite proxy by making API calls to `/api/*` instead of `http://localhost:5000/*`
- **Styling Issues**: Verify Tailwind CSS is properly configured
- **API Connection**: Confirm the backend server is running on port 5000

## 📄 License

This project is part of the AI-Driven Stress Management System for educational purposes.