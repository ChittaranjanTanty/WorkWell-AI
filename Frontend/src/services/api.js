import axios from 'axios';

const API_BASE_URL = '/api'; // Using proxy

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API Service methods
export const apiService = {
  // Health check
  checkHealth: async () => {
    const response = await api.get('/health');
    return response.data;
  },

  // Employee management
  registerEmployee: async (employeeData) => {
    const response = await api.post('/employee/register', employeeData);
    return response.data;
  },

  getEmployee: async (employeeId) => {
    const response = await api.get(`/employee/${employeeId}`);
    return response.data;
  },

  getAllEmployees: async () => {
    const response = await api.get('/employees');
    return response.data;
  },

  deleteEmployee: async (employeeId) => {
    const response = await api.delete(`/employee/${employeeId}`);
    return response.data;
  },

  // Predictions
  predictDemo: async (data) => {
    const response = await api.post('/predict/demo', data);
    return response.data;
  },

  predictEmployee: async (data) => {
    const response = await api.post('/predict/employee', data);
    return response.data;
  },

  // Alerts
  getAlerts: async () => {
    const response = await api.get('/alerts');
    return response.data;
  },

  // Model Analytics
  getModelAnalytics: async () => {
    const response = await api.get('/model/analytics');
    return response.data;
  },
};

export default api;
