import { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  BarChart3, 
  Activity, 
  Target, 
  Award,
  Brain,
  Layers,
  Users,
  AlertCircle,
  CheckCircle,
  TrendingDown,
  RefreshCw
} from 'lucide-react';
import { apiService } from '../services/api';

function Analytics() {
  const [analytics, setAnalytics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchAnalytics = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiService.getModelAnalytics();
      setAnalytics(data);
    } catch (err) {
      setError('Failed to load analytics data');
      console.error('Error fetching analytics:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAnalytics();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <RefreshCw className="h-12 w-12 mx-auto text-indigo-600 animate-spin mb-4" />
          <p className="text-gray-600">Loading model analytics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card bg-red-50 border-red-200">
        <div className="flex items-center gap-3">
          <AlertCircle className="h-6 w-6 text-red-600" />
          <div>
            <h3 className="text-lg font-semibold text-red-900">{error}</h3>
            <button 
              onClick={fetchAnalytics}
              className="text-sm text-red-600 hover:text-red-700 underline mt-1"
            >
              Try again
            </button>
          </div>
        </div>
      </div>
    );
  }

  const { model_info, performance_metrics, feature_importance, class_distribution, session_statistics } = analytics || {};

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h2 className="text-3xl font-bold text-gray-900">Model Analytics</h2>
          <p className="text-gray-600 mt-1">Comprehensive model performance and insights</p>
        </div>
        <button
          onClick={fetchAnalytics}
          className="btn btn-secondary flex items-center gap-2"
        >
          <RefreshCw className="h-4 w-4" />
          Refresh
        </button>
      </div>

      {/* Model Info Section */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="card bg-gradient-to-br from-indigo-500 to-indigo-600 text-white">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-indigo-100 text-sm">Model Type</p>
              <h3 className="text-2xl font-bold mt-1">{model_info?.model_type?.split(' ')[0]}</h3>
              <p className="text-indigo-100 text-xs mt-1">{model_info?.framework}</p>
            </div>
            <Brain className="h-8 w-8 text-indigo-200" />
          </div>
        </div>

        <div className="card bg-gradient-to-br from-green-500 to-green-600 text-white">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-green-100 text-sm">Accuracy</p>
              <h3 className="text-2xl font-bold mt-1">{performance_metrics?.accuracy}%</h3>
              <p className="text-green-100 text-xs mt-1">Overall Performance</p>
            </div>
            <Target className="h-8 w-8 text-green-200" />
          </div>
        </div>

        <div className="card bg-gradient-to-br from-purple-500 to-purple-600 text-white">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-purple-100 text-sm">F1 Score</p>
              <h3 className="text-2xl font-bold mt-1">{performance_metrics?.f1_score}%</h3>
              <p className="text-purple-100 text-xs mt-1">Balanced Metric</p>
            </div>
            <Award className="h-8 w-8 text-purple-200" />
          </div>
        </div>

        <div className="card bg-gradient-to-br from-orange-500 to-orange-600 text-white">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-orange-100 text-sm">Features</p>
              <h3 className="text-2xl font-bold mt-1">{model_info?.n_features}</h3>
              <p className="text-orange-100 text-xs mt-1">{model_info?.n_classes} Classes</p>
            </div>
            <Layers className="h-8 w-8 text-orange-200" />
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Metrics Details */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-indigo-600" />
            Performance Metrics
          </h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-600">Accuracy</span>
                <span className="font-semibold text-gray-900">{performance_metrics?.accuracy}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-green-600 h-2 rounded-full transition-all duration-500" 
                  style={{ width: `${performance_metrics?.accuracy}%` }}
                ></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-600">Precision</span>
                <span className="font-semibold text-gray-900">{performance_metrics?.precision}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-500" 
                  style={{ width: `${performance_metrics?.precision}%` }}
                ></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-600">Recall</span>
                <span className="font-semibold text-gray-900">{performance_metrics?.recall}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-purple-600 h-2 rounded-full transition-all duration-500" 
                  style={{ width: `${performance_metrics?.recall}%` }}
                ></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-gray-600">F1 Score</span>
                <span className="font-semibold text-gray-900">{performance_metrics?.f1_score}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-indigo-600 h-2 rounded-full transition-all duration-500" 
                  style={{ width: `${performance_metrics?.f1_score}%` }}
                ></div>
              </div>
            </div>

            <div className="pt-4 border-t border-gray-200">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-gray-600">Training Samples</p>
                  <p className="font-semibold text-gray-900">{performance_metrics?.training_samples?.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-gray-600">Test Samples</p>
                  <p className="font-semibold text-gray-900">{performance_metrics?.test_samples?.toLocaleString()}</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Feature Importance */}
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Activity className="h-5 w-5 text-indigo-600" />
            Top Feature Importance
          </h3>
          <div className="space-y-3">
            {feature_importance?.names?.slice(0, 10).map((name, index) => (
              <div key={index}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-700 truncate" title={name}>{name}</span>
                  <span className="font-semibold text-gray-900 ml-2">
                    {feature_importance?.percentages[index]?.toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-1.5">
                  <div 
                    className="bg-gradient-to-r from-indigo-500 to-purple-500 h-1.5 rounded-full transition-all duration-500" 
                    style={{ width: `${(feature_importance?.percentages[index] / Math.max(...feature_importance?.percentages)) * 100}%` }}
                  ></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Session Statistics */}
      {session_statistics?.total_predictions > 0 && (
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
            <Users className="h-5 w-5 text-indigo-600" />
            Current Session Statistics
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-3xl font-bold text-indigo-600">
                {session_statistics?.total_predictions}
              </div>
              <div className="text-sm text-gray-600 mt-1">Total Assessments</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-green-600">
                {session_statistics?.total_employees}
              </div>
              <div className="text-sm text-gray-600 mt-1">Employees Monitored</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-orange-600">
                {(session_statistics?.average_stress_score * 100)?.toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600 mt-1">Avg Stress Level</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-red-600">
                {session_statistics?.alert_distribution?.CRITICAL || 0}
              </div>
              <div className="text-sm text-gray-600 mt-1">Critical Alerts</div>
            </div>
          </div>

          {/* Alert Distribution */}
          <div className="mt-6 pt-6 border-t border-gray-200">
            <h4 className="text-sm font-semibold text-gray-700 mb-3">Alert Level Distribution</h4>
            <div className="space-y-2">
              <div className="flex items-center gap-3">
                <CheckCircle className="h-5 w-5 text-green-600" />
                <div className="flex-1">
                  <div className="flex justify-between text-sm mb-1">
                    <span>Normal</span>
                    <span className="font-semibold">{session_statistics?.alert_distribution?.NORMAL || 0}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-green-600 h-2 rounded-full" 
                      style={{ 
                        width: `${(session_statistics?.alert_distribution?.NORMAL / session_statistics?.total_predictions * 100) || 0}%` 
                      }}
                    ></div>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <AlertCircle className="h-5 w-5 text-yellow-600" />
                <div className="flex-1">
                  <div className="flex justify-between text-sm mb-1">
                    <span>Warning</span>
                    <span className="font-semibold">{session_statistics?.alert_distribution?.WARNING || 0}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-yellow-600 h-2 rounded-full" 
                      style={{ 
                        width: `${(session_statistics?.alert_distribution?.WARNING / session_statistics?.total_predictions * 100) || 0}%` 
                      }}
                    ></div>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <AlertCircle className="h-5 w-5 text-red-600" />
                <div className="flex-1">
                  <div className="flex justify-between text-sm mb-1">
                    <span>Critical</span>
                    <span className="font-semibold">{session_statistics?.alert_distribution?.CRITICAL || 0}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-red-600 h-2 rounded-full" 
                      style={{ 
                        width: `${(session_statistics?.alert_distribution?.CRITICAL / session_statistics?.total_predictions * 100) || 0}%` 
                      }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Class Distribution */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-indigo-600" />
          Training Data Class Distribution
        </h3>
        <div className="space-y-3">
          {class_distribution?.classes?.map((className, index) => {
            const count = class_distribution?.counts[index];
            const percentage = class_distribution?.percentages[index];
            if (count === 0) return null;
            
            return (
              <div key={index}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-700">{className}</span>
                  <span className="font-semibold text-gray-900">
                    {count?.toLocaleString()} ({percentage}%)
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-indigo-600 h-2 rounded-full transition-all duration-500" 
                    style={{ width: `${percentage}%` }}
                  ></div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Model Comparison Section */}
      <div className="card bg-gradient-to-br from-gray-50 to-gray-100">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-indigo-600" />
          Model Evaluation & Selection
        </h3>
        <div className="bg-white rounded-lg p-6 shadow-sm">
          <p className="text-gray-600 mb-6">
            Multiple machine learning models were evaluated for stress detection. The comparison below shows why Random Forest was selected as the production model.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Random Forest - Winner */}
            <div className="border-2 border-green-500 rounded-lg p-4 bg-green-50 relative">
              <div className="absolute -top-3 left-4 bg-green-500 text-white px-3 py-1 rounded-full text-xs font-bold">
                ✓ SELECTED MODEL
              </div>
              <h4 className="text-lg font-bold text-gray-900 mt-2 mb-3">Random Forest</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-600">Accuracy:</span>
                  <span className="font-bold text-green-700">83.22%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Precision:</span>
                  <span className="font-bold text-green-700">82.13%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Recall:</span>
                  <span className="font-bold text-green-700">83.22%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">F1-Score:</span>
                  <span className="font-bold text-green-700">82.61%</span>
                </div>
              </div>
              <div className="mt-4 pt-4 border-t border-green-200">
                <p className="text-sm text-gray-700">
                  <strong>Why chosen:</strong> Excellent balance of accuracy and performance. Handles multi-class classification well with consistent predictions.
                </p>
              </div>
            </div>

            {/* SVM - Not Selected */}
            <div className="border border-gray-300 rounded-lg p-4 bg-gray-50 opacity-75">
              <h4 className="text-lg font-bold text-gray-900 mb-3">Support Vector Machine (SVM)</h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-600">Accuracy:</span>
                  <span className="font-semibold text-gray-700">13.36%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Precision:</span>
                  <span className="font-semibold text-gray-700">20.39%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Recall:</span>
                  <span className="font-semibold text-gray-700">13.36%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">F1-Score:</span>
                  <span className="font-semibold text-gray-700">12.91%</span>
                </div>
              </div>
              <div className="mt-4 pt-4 border-t border-gray-300">
                <p className="text-sm text-gray-600">
                  <strong>Not selected:</strong> Poor performance on 8-class WESAD dataset. Struggles with multi-class stress state classification.
                </p>
              </div>
            </div>
          </div>

          <div className="mt-6 bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
            <div className="flex gap-3">
              <CheckCircle className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
              <div>
                <h5 className="font-semibold text-blue-900 mb-1">Training Details</h5>
                <p className="text-sm text-blue-800">
                  Both models were trained on <strong>4,328 samples</strong> (after SMOTE balancing) and tested on <strong>292 samples</strong>. 
                  Random Forest achieved <strong>6.4x better accuracy</strong> than SVM, making it the clear choice for production deployment.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Model Details */}
      <div className="card bg-gray-50">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Model Details</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <p className="text-sm text-gray-600">Framework</p>
            <p className="text-lg font-semibold text-gray-900">{model_info?.framework}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Model Type</p>
            <p className="text-lg font-semibold text-gray-900">{model_info?.model_type}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Dataset</p>
            <p className="text-lg font-semibold text-gray-900">{model_info?.trained_on}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Number of Estimators</p>
            <p className="text-lg font-semibold text-gray-900">{model_info?.n_estimators}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Input Features</p>
            <p className="text-lg font-semibold text-gray-900">{model_info?.n_features}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Output Classes</p>
            <p className="text-lg font-semibold text-gray-900">{model_info?.n_classes}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Analytics;
