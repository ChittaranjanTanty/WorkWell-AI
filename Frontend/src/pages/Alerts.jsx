import { useState, useEffect } from 'react';
import { AlertTriangle, RefreshCw, Bell, TrendingUp } from 'lucide-react';
import { apiService } from '../services/api';

function Alerts() {
  const [alerts, setAlerts] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAlerts();
  }, []);

  const fetchAlerts = async () => {
    setLoading(true);
    try {
      const data = await apiService.getAlerts();
      setAlerts(data.alerts || []);
    } catch (error) {
      console.error('Error fetching alerts:', error);
    } finally {
      setLoading(false);
    }
  };

  const getAlertIcon = (level) => {
    return level === 'CRITICAL' ? '🔴' : '🟡';
  };

  const getAlertClass = (level) => {
    return level === 'CRITICAL'
      ? 'bg-red-50 border-red-200'
      : 'bg-yellow-50 border-yellow-200';
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold text-gray-900">Active Alerts</h2>
          <p className="text-gray-600 mt-1">Monitor critical and warning stress levels</p>
        </div>
        <button
          onClick={fetchAlerts}
          disabled={loading}
          className="btn btn-primary flex items-center space-x-2"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      <div className="card">
        {loading ? (
          <div className="text-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
            <p className="text-gray-500 mt-4">Loading alerts...</p>
          </div>
        ) : alerts.length === 0 ? (
          <div className="text-center py-12">
            <Bell className="h-16 w-16 mx-auto text-gray-300 mb-4" />
            <p className="text-gray-500">No active alerts</p>
            <p className="text-sm text-gray-400 mt-2">All employees are within normal stress levels</p>
          </div>
        ) : (
          <div className="space-y-4">
            {alerts.map((alert, index) => (
              <div
                key={index}
                className={`p-6 rounded-lg border-2 ${getAlertClass(alert.alert_level)}`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-4">
                    <span className="text-3xl">{getAlertIcon(alert.alert_level)}</span>
                    <div>
                      <h4 className="font-bold text-gray-900 text-lg">
                        {alert.name || alert.employee_id}
                      </h4>
                      <p className="text-sm text-gray-600">{alert.department}</p>
                      <div className="mt-2 space-y-1">
                        <p className="text-sm">
                          <span className="font-medium">Employee ID:</span> {alert.employee_id}
                        </p>
                        <p className="text-sm">
                          <span className="font-medium">Stress Score:</span>{' '}
                          <span className="font-bold">{(alert.stress_score * 100).toFixed(1)}%</span>
                        </p>
                        <p className="text-xs text-gray-500">{alert.timestamp}</p>
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-col items-end">
                    <span
                      className={`px-3 py-1 rounded-full text-sm font-bold ${
                        alert.alert_level === 'CRITICAL'
                          ? 'bg-red-200 text-red-800'
                          : 'bg-yellow-200 text-yellow-800'
                      }`}
                    >
                      {alert.alert_level}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default Alerts;
