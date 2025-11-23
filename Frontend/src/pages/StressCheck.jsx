import { useState } from 'react';
import { Heart, Activity, Thermometer, Wind, TrendingUp, Send, AlertCircle, CheckCircle } from 'lucide-react';
import { apiService } from '../services/api';

function StressCheck() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [formData, setFormData] = useState({
    employee_id: '',
    name: '',
    age: '',
    department: '',
    physiological_data: {
      heart_rate: '',
      hrv_mean: '',
      eda_mean: '',
      eda_std: '',
      temperature: '',
      respiration_rate: '',
      activity_level: '',
    },
    context: {
      time_of_day: 'afternoon',
      workload: 'medium',
      meeting_scheduled: false,
      deadline_approaching: false,
    },
  });

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    
    if (name.startsWith('physio_')) {
      const key = name.replace('physio_', '');
      setFormData({
        ...formData,
        physiological_data: {
          ...formData.physiological_data,
          [key]: value,
        },
      });
    } else if (name.startsWith('context_')) {
      const key = name.replace('context_', '');
      setFormData({
        ...formData,
        context: {
          ...formData.context,
          [key]: type === 'checkbox' ? checked : value,
        },
      });
    } else {
      setFormData({
        ...formData,
        [name]: value,
      });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      // Convert string values to numbers for physiological data
      const processedData = {
        ...formData,
        age: parseInt(formData.age) || 30,
        physiological_data: {
          heart_rate: parseFloat(formData.physiological_data.heart_rate) || 75,
          hrv_mean: parseFloat(formData.physiological_data.hrv_mean) || 50,
          eda_mean: parseFloat(formData.physiological_data.eda_mean) || 0.5,
          eda_std: parseFloat(formData.physiological_data.eda_std) || 0.1,
          temperature: parseFloat(formData.physiological_data.temperature) || 36.5,
          respiration_rate: parseFloat(formData.physiological_data.respiration_rate) || 16,
          activity_level: parseFloat(formData.physiological_data.activity_level) || 0.3,
        },
      };

      const response = await apiService.predictEmployee(processedData);
      setResult(response);
    } catch (error) {
      console.error('Error predicting stress:', error);
      alert('Error: ' + (error.response?.data?.error || error.message));
    } finally {
      setLoading(false);
    }
  };

  const getAlertColorClass = (alertLevel) => {
    switch (alertLevel) {
      case 'CRITICAL':
        return 'bg-red-100 border-red-300 text-red-800';
      case 'WARNING':
        return 'bg-yellow-100 border-yellow-300 text-yellow-800';
      default:
        return 'bg-green-100 border-green-300 text-green-800';
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-gray-900">Stress Assessment</h2>
        <p className="text-gray-600 mt-1">Comprehensive employee stress evaluation</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Form */}
        <div className="card">
          <h3 className="text-xl font-bold text-gray-900 mb-6">Employee Information</h3>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Basic Info */}
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Employee ID *
                </label>
                <input
                  type="text"
                  name="employee_id"
                  value={formData.employee_id}
                  onChange={handleInputChange}
                  className="input"
                  placeholder="EMP001"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Name
                </label>
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleInputChange}
                  className="input"
                  placeholder="John Doe"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Age
                  </label>
                  <input
                    type="number"
                    name="age"
                    value={formData.age}
                    onChange={handleInputChange}
                    className="input"
                    placeholder="35"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Department
                  </label>
                  <select
                    name="department"
                    value={formData.department}
                    onChange={handleInputChange}
                    className="input"
                  >
                    <option value="">Select...</option>
                    <option value="Engineering">Engineering</option>
                    <option value="Sales">Sales</option>
                    <option value="Marketing">Marketing</option>
                    <option value="HR">HR</option>
                    <option value="Finance">Finance</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Physiological Data */}
            <div className="border-t pt-6">
              <h4 className="font-semibold text-gray-900 mb-4 flex items-center">
                <Activity className="h-5 w-5 mr-2 text-blue-600" />
                Physiological Readings
              </h4>
              
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Heart Rate (BPM)
                  </label>
                  <input
                    type="number"
                    name="physio_heart_rate"
                    value={formData.physiological_data.heart_rate}
                    onChange={handleInputChange}
                    className="input"
                    placeholder="75"
                    step="0.1"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    HRV (ms)
                  </label>
                  <input
                    type="number"
                    name="physio_hrv_mean"
                    value={formData.physiological_data.hrv_mean}
                    onChange={handleInputChange}
                    className="input"
                    placeholder="50"
                    step="0.1"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    EDA Mean (µS)
                  </label>
                  <input
                    type="number"
                    name="physio_eda_mean"
                    value={formData.physiological_data.eda_mean}
                    onChange={handleInputChange}
                    className="input"
                    placeholder="0.5"
                    step="0.01"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Temperature (°C)
                  </label>
                  <input
                    type="number"
                    name="physio_temperature"
                    value={formData.physiological_data.temperature}
                    onChange={handleInputChange}
                    className="input"
                    placeholder="36.5"
                    step="0.1"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Respiration (bpm)
                  </label>
                  <input
                    type="number"
                    name="physio_respiration_rate"
                    value={formData.physiological_data.respiration_rate}
                    onChange={handleInputChange}
                    className="input"
                    placeholder="16"
                    step="0.1"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Activity Level (0-1)
                  </label>
                  <input
                    type="number"
                    name="physio_activity_level"
                    value={formData.physiological_data.activity_level}
                    onChange={handleInputChange}
                    className="input"
                    placeholder="0.3"
                    step="0.1"
                    min="0"
                    max="1"
                  />
                </div>
              </div>
            </div>

            {/* Context */}
            <div className="border-t pt-6">
              <h4 className="font-semibold text-gray-900 mb-4">Work Context</h4>
              
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Time of Day
                    </label>
                    <select
                      name="context_time_of_day"
                      value={formData.context.time_of_day}
                      onChange={handleInputChange}
                      className="input"
                    >
                      <option value="morning">Morning</option>
                      <option value="afternoon">Afternoon</option>
                      <option value="evening">Evening</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Workload
                    </label>
                    <select
                      name="context_workload"
                      value={formData.context.workload}
                      onChange={handleInputChange}
                      className="input"
                    >
                      <option value="low">Low</option>
                      <option value="medium">Medium</option>
                      <option value="high">High</option>
                    </select>
                  </div>
                </div>

                <div className="flex items-center space-x-6">
                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      name="context_meeting_scheduled"
                      checked={formData.context.meeting_scheduled}
                      onChange={handleInputChange}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="text-sm text-gray-700">Meeting Scheduled</span>
                  </label>

                  <label className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      name="context_deadline_approaching"
                      checked={formData.context.deadline_approaching}
                      onChange={handleInputChange}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="text-sm text-gray-700">Deadline Approaching</span>
                  </label>
                </div>
              </div>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="btn btn-primary w-full flex items-center justify-center space-x-2"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <Send className="h-5 w-5" />
                  <span>Assess Stress Level</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Results */}
        <div className="space-y-6">
          {result ? (
            <>
              {/* Alert Card */}
              <div className={`card border-2 ${getAlertColorClass(result.stress_assessment?.alert_level)}`}>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-bold">Stress Assessment Result</h3>
                  <span className={`px-3 py-1 rounded-full text-sm font-bold ${
                    result.stress_assessment?.alert_level === 'CRITICAL' ? 'bg-red-200' :
                    result.stress_assessment?.alert_level === 'WARNING' ? 'bg-yellow-200' :
                    'bg-green-200'
                  }`}>
                    {result.stress_assessment?.alert_level}
                  </span>
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Stress Score:</span>
                    <span className="text-2xl font-bold">{result.stress_assessment?.stress_percentage}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Risk Level:</span>
                    <span className="text-sm">{result.stress_assessment?.risk_level}</span>
                  </div>
                </div>
              </div>

              {/* Physiological Analysis */}
              <div className="card">
                <h4 className="font-bold text-gray-900 mb-4">Physiological Indicators</h4>
                <div className="space-y-3">
                  {result.physiological_analysis && Object.entries(result.physiological_analysis).map(([key, data]) => (
                    <div key={key} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center space-x-2">
                        <span className="text-2xl">{data.indicator}</span>
                        <div>
                          <p className="font-medium capitalize">{key.replace('_', ' ')}</p>
                          <p className="text-sm text-gray-600">{data.status}</p>
                        </div>
                      </div>
                      <span className="font-bold text-gray-900">{data.value}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Recommendations */}
              <div className="card">
                <h4 className="font-bold text-gray-900 mb-4 flex items-center">
                  <CheckCircle className="h-5 w-5 mr-2 text-green-600" />
                  Personalized Recommendations
                </h4>
                <ul className="space-y-2">
                  {result.recommendations?.map((rec, index) => (
                    <li key={index} className="flex items-start space-x-2 text-sm">
                      <span className="text-blue-600 mt-1">•</span>
                      <span>{rec}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Suggested Actions */}
              {result.suggested_actions && (
                <div className="card">
                  <h4 className="font-bold text-gray-900 mb-4">Action Plan</h4>
                  <div className="space-y-4">
                    <div>
                      <p className="text-sm font-semibold text-gray-700 mb-2">Immediate Actions:</p>
                      <ul className="text-sm space-y-1 ml-4">
                        {result.suggested_actions.immediate?.slice(0, 3).map((action, idx) => (
                          <li key={idx} className="text-gray-600">• {action}</li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-gray-700 mb-2">Short-term (Today/This Week):</p>
                      <ul className="text-sm space-y-1 ml-4">
                        {result.suggested_actions.short_term?.slice(0, 2).map((action, idx) => (
                          <li key={idx} className="text-gray-600">• {action}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              )}

              {/* Follow-up */}
              {result.follow_up && (
                <div className="card bg-blue-50">
                  <div className="flex items-start space-x-3">
                    <AlertCircle className="h-5 w-5 text-blue-600 mt-0.5" />
                    <div>
                      <p className="font-semibold text-blue-900">Follow-up Required</p>
                      <p className="text-sm text-blue-700 mt-1">
                        Next check-in: {result.follow_up.next_check_in}
                      </p>
                      {result.follow_up.escalation_required && (
                        <p className="text-sm text-red-700 mt-1 font-medium">
                          ⚠️ Escalation to management required
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="card text-center py-12">
              <Heart className="h-16 w-16 mx-auto text-gray-300 mb-4" />
              <p className="text-gray-500">
                Fill in the form and submit to see stress assessment results
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default StressCheck;
