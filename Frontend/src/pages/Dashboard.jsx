import { useState, useEffect } from 'react';
import { Users, AlertTriangle, Activity, TrendingUp, RefreshCw } from 'lucide-react';
import { apiService } from '../services/api';
import StatsCard from '../components/StatsCard';
import AlertsList from '../components/AlertsList';

function Dashboard() {
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState({
    totalEmployees: 0,
    activeAlerts: 0,
    avgStressLevel: 0,
    healthStatus: 'checking',
  });
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    setLoading(true);
    try {
      // Fetch health status
      const health = await apiService.checkHealth();
      
      // Fetch employees
      const employeesData = await apiService.getAllEmployees();
      
      // Fetch alerts
      const alertsData = await apiService.getAlerts();

      // Calculate average stress
      const avgStress = employeesData.employees?.reduce((sum, emp) => 
        sum + (emp.average_stress || 0), 0
      ) / (employeesData.employees?.length || 1);

      setStats({
        totalEmployees: employeesData.total_employees || 0,
        activeAlerts: alertsData.total_alerts || 0,
        avgStressLevel: avgStress * 100,
        healthStatus: health.status,
      });

      setAlerts(alertsData.alerts || []);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold text-gray-900">Dashboard</h2>
          <p className="text-gray-600 mt-1">Real-time employee wellness monitoring</p>
        </div>
        <button
          onClick={fetchDashboardData}
          disabled={loading}
          className="btn btn-primary flex items-center space-x-2"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Total Employees"
          value={stats.totalEmployees}
          icon={Users}
          color="blue"
          loading={loading}
        />
        <StatsCard
          title="Active Alerts"
          value={stats.activeAlerts}
          icon={AlertTriangle}
          color="red"
          loading={loading}
        />
        <StatsCard
          title="Avg Stress Level"
          value={`${stats.avgStressLevel.toFixed(1)}%`}
          icon={Activity}
          color="purple"
          loading={loading}
        />
        <StatsCard
          title="System Status"
          value={stats.healthStatus === 'healthy' ? 'Healthy' : 'Checking'}
          icon={TrendingUp}
          color="green"
          loading={loading}
        />
      </div>

      {/* Recent Alerts */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold text-gray-900">Recent Alerts</h3>
          <span className="text-sm text-gray-500">
            {alerts.length} active alert{alerts.length !== 1 ? 's' : ''}
          </span>
        </div>
        <AlertsList alerts={alerts} loading={loading} />
      </div>

      {/* Quick Actions */}
      <div className="card">
        <h3 className="text-xl font-bold text-gray-900 mb-4">Quick Actions</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <button className="p-6 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-500 hover:bg-blue-50 transition-all text-center">
            <Activity className="h-8 w-8 mx-auto mb-2 text-blue-600" />
            <p className="font-medium text-gray-900">Quick Stress Check</p>
            <p className="text-sm text-gray-500 mt-1">Run instant assessment</p>
          </button>
          <button className="p-6 border-2 border-dashed border-gray-300 rounded-lg hover:border-green-500 hover:bg-green-50 transition-all text-center">
            <Users className="h-8 w-8 mx-auto mb-2 text-green-600" />
            <p className="font-medium text-gray-900">Register Employee</p>
            <p className="text-sm text-gray-500 mt-1">Add new team member</p>
          </button>
          <button className="p-6 border-2 border-dashed border-gray-300 rounded-lg hover:border-purple-500 hover:bg-purple-50 transition-all text-center">
            <TrendingUp className="h-8 w-8 mx-auto mb-2 text-purple-600" />
            <p className="font-medium text-gray-900">View Analytics</p>
            <p className="text-sm text-gray-500 mt-1">Detailed insights</p>
          </button>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
