import { AlertTriangle, Bell } from 'lucide-react';

function AlertsList({ alerts, loading }) {
  if (loading) {
    return (
      <div className="space-y-3">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-20 bg-gray-200 animate-pulse rounded-lg"></div>
        ))}
      </div>
    );
  }

  if (!alerts || alerts.length === 0) {
    return (
      <div className="text-center py-8">
        <Bell className="h-12 w-12 mx-auto text-gray-300 mb-2" />
        <p className="text-gray-500 text-sm">No active alerts</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {alerts.slice(0, 5).map((alert, index) => (
        <div
          key={index}
          className={`p-4 rounded-lg border-l-4 ${
            alert.alert_level === 'CRITICAL'
              ? 'bg-red-50 border-red-500'
              : 'bg-yellow-50 border-yellow-500'
          }`}
        >
          <div className="flex items-start justify-between">
            <div className="flex items-start space-x-3">
              <AlertTriangle
                className={`h-5 w-5 mt-0.5 ${
                  alert.alert_level === 'CRITICAL' ? 'text-red-600' : 'text-yellow-600'
                }`}
              />
              <div>
                <p className="font-medium text-gray-900">
                  {alert.name || alert.employee_id}
                </p>
                <p className="text-sm text-gray-600">{alert.department}</p>
                <p className="text-xs text-gray-500 mt-1">
                  Stress: {(alert.stress_score * 100).toFixed(1)}%
                </p>
              </div>
            </div>
            <span
              className={`px-2 py-1 text-xs font-semibold rounded ${
                alert.alert_level === 'CRITICAL'
                  ? 'bg-red-200 text-red-800'
                  : 'bg-yellow-200 text-yellow-800'
              }`}
            >
              {alert.alert_level}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

export default AlertsList;
