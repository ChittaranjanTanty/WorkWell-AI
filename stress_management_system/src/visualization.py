"""
Visualization and Escalation Module
Creates dashboards and handles stress escalation alerts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class StressVisualization:
    """
    Visualization and escalation system for stress monitoring
    """
    
    def __init__(self, stress_threshold=0.7, escalation_threshold=0.85):
        """
        Initialize visualization system
        
        Args:
            stress_threshold: Threshold for stress alert
            escalation_threshold: Threshold for critical escalation
        """
        self.stress_threshold = stress_threshold
        self.escalation_threshold = escalation_threshold
        self.escalation_log = []
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def check_escalation(self, stress_probability, timestamp=None):
        """
        Check if stress level requires escalation
        
        Args:
            stress_probability: Probability of stress (0-1)
            timestamp: Timestamp of measurement
            
        Returns:
            Escalation level and message
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        escalation_level = "NORMAL"
        message = ""
        
        if stress_probability >= self.escalation_threshold:
            escalation_level = "CRITICAL"
            message = "🚨 CRITICAL STRESS ALERT: Immediate intervention recommended!"
            self._log_escalation(timestamp, stress_probability, escalation_level)
        elif stress_probability >= self.stress_threshold:
            escalation_level = "WARNING"
            message = "⚠️ ELEVATED STRESS: Stress management techniques recommended."
            self._log_escalation(timestamp, stress_probability, escalation_level)
        else:
            message = "✅ Normal stress levels."
        
        return escalation_level, message
    
    def _log_escalation(self, timestamp, stress_level, escalation_level):
        """Log escalation event"""
        self.escalation_log.append({
            'timestamp': timestamp,
            'stress_level': stress_level,
            'escalation_level': escalation_level
        })
    
    def plot_stress_probability(self, y_pred_proba, y_true=None, save_path=None):
        """
        Plot stress probability distribution
        
        Args:
            y_pred_proba: Predicted probabilities (n_samples, n_classes)
            y_true: True labels (optional)
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Extract stress probabilities (class 1)
        if y_pred_proba.ndim > 1:
            stress_probs = y_pred_proba[:, 1]
        else:
            stress_probs = y_pred_proba
        
        # Histogram
        axes[0].hist(stress_probs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(self.stress_threshold, color='orange', linestyle='--', 
                       linewidth=2, label=f'Stress Threshold ({self.stress_threshold})')
        axes[0].axvline(self.escalation_threshold, color='red', linestyle='--', 
                       linewidth=2, label=f'Escalation Threshold ({self.escalation_threshold})')
        axes[0].set_xlabel('Stress Probability')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Stress Probabilities')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_probs = np.sort(stress_probs)
        cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
        axes[1].plot(sorted_probs, cumulative, linewidth=2)
        axes[1].axvline(self.stress_threshold, color='orange', linestyle='--', 
                       linewidth=2, label=f'Stress Threshold')
        axes[1].axvline(self.escalation_threshold, color='red', linestyle='--', 
                       linewidth=2, label=f'Escalation Threshold')
        axes[1].set_xlabel('Stress Probability')
        axes[1].set_ylabel('Cumulative Proportion')
        axes[1].set_title('Cumulative Distribution of Stress Probabilities')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        high_stress_count = np.sum(stress_probs >= self.stress_threshold)
        critical_count = np.sum(stress_probs >= self.escalation_threshold)
        
        print(f"\nStress Probability Statistics:")
        print(f"Mean: {np.mean(stress_probs):.3f}")
        print(f"Std: {np.std(stress_probs):.3f}")
        print(f"High stress cases (≥{self.stress_threshold}): {high_stress_count} ({high_stress_count/len(stress_probs)*100:.1f}%)")
        print(f"Critical cases (≥{self.escalation_threshold}): {critical_count} ({critical_count/len(stress_probs)*100:.1f}%)")
    
    def plot_realtime_monitor(self, stress_history, timestamps=None, save_path=None):
        """
        Plot real-time stress monitoring dashboard
        
        Args:
            stress_history: List of stress probabilities over time
            timestamps: List of timestamps
            save_path: Path to save plot
        """
        if timestamps is None:
            timestamps = list(range(len(stress_history)))
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Real-Time Stress Level', 'Stress Level Distribution'),
            row_heights=[0.6, 0.4],
            vertical_spacing=0.12
        )
        
        # Time series
        colors = ['green' if s < self.stress_threshold else 'orange' if s < self.escalation_threshold else 'red' 
                  for s in stress_history]
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=stress_history,
                mode='lines+markers',
                name='Stress Level',
                line=dict(color='blue', width=2),
                marker=dict(size=6, color=colors)
            ),
            row=1, col=1
        )
        
        # Threshold lines
        fig.add_hline(
            y=self.stress_threshold,
            line_dash="dash",
            line_color="orange",
            annotation_text="Stress Threshold",
            row=1, col=1
        )
        
        fig.add_hline(
            y=self.escalation_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="Critical Threshold",
            row=1, col=1
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=stress_history,
                nbinsx=30,
                name='Distribution',
                marker_color='skyblue'
            ),
            row=2, col=1
        )
        
        # Layout
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Stress Probability", row=2, col=1)
        fig.update_yaxes(title_text="Stress Probability", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Real-Time Stress Monitoring Dashboard",
            title_font_size=20
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def create_session_report(self, session_data, save_path=None):
        """
        Create comprehensive session report
        
        Args:
            session_data: Dictionary containing session information
                - predictions: Predicted labels
                - probabilities: Prediction probabilities
                - true_labels: True labels (optional)
                - explanations: XAI explanations
                - recommendations: GenAI recommendations
            save_path: Path to save report
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Prediction Distribution',
                'Stress Probability by Class',
                'Confidence Distribution',
                'Escalation Status',
                'Top Contributing Features',
                'Session Summary'
            ),
            specs=[
                [{"type": "bar"}, {"type": "box"}],
                [{"type": "histogram"}, {"type": "indicator"}],
                [{"type": "bar"}, {"type": "table"}]
            ],
            row_heights=[0.3, 0.3, 0.4],
            vertical_spacing=0.1
        )
        
        predictions = session_data['predictions']
        probabilities = session_data['probabilities']
        
        # 1. Prediction distribution
        unique, counts = np.unique(predictions, return_counts=True)
        labels = ['Baseline', 'Stress', 'Amusement']
        pred_labels = [labels[i] for i in unique]
        
        fig.add_trace(
            go.Bar(x=pred_labels, y=counts, marker_color=['green', 'red', 'blue'][:len(unique)]),
            row=1, col=1
        )
        
        # 2. Stress probability box plot
        if probabilities.ndim > 1:
            for i, label in enumerate(['Baseline', 'Stress', 'Amusement']):
                fig.add_trace(
                    go.Box(y=probabilities[:, i], name=label),
                    row=1, col=2
                )
        
        # 3. Confidence distribution
        max_probs = np.max(probabilities, axis=1) if probabilities.ndim > 1 else probabilities
        fig.add_trace(
            go.Histogram(x=max_probs, nbinsx=30, marker_color='purple'),
            row=2, col=1
        )
        
        # 4. Escalation indicator
        stress_probs = probabilities[:, 1] if probabilities.ndim > 1 else probabilities
        escalation_count = np.sum(stress_probs >= self.escalation_threshold)
        warning_count = np.sum((stress_probs >= self.stress_threshold) & (stress_probs < self.escalation_threshold))
        
        escalation_status = "CRITICAL" if escalation_count > 0 else "WARNING" if warning_count > 0 else "NORMAL"
        status_color = "red" if escalation_status == "CRITICAL" else "orange" if escalation_status == "WARNING" else "green"
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=np.mean(stress_probs) * 100,
                title={'text': f"Avg Stress Level<br><span style='font-size:0.8em;color:{status_color}'>{escalation_status}</span>"},
                delta={'reference': self.stress_threshold * 100},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': status_color},
                    'steps': [
                        {'range': [0, self.stress_threshold * 100], 'color': 'lightgreen'},
                        {'range': [self.stress_threshold * 100, self.escalation_threshold * 100], 'color': 'lightyellow'},
                        {'range': [self.escalation_threshold * 100, 100], 'color': 'lightcoral'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': self.escalation_threshold * 100
                    }
                }
            ),
            row=2, col=2
        )
        
        # 5. Top features (if available)
        if 'top_features' in session_data:
            features = list(session_data['top_features'].keys())[:10]
            importances = [session_data['top_features'][f] for f in features]
            
            fig.add_trace(
                go.Bar(x=importances, y=features, orientation='h', marker_color='teal'),
                row=3, col=1
            )
        
        # 6. Summary table
        summary_data = {
            'Metric': ['Total Samples', 'Stress Cases', 'Critical Cases', 'Avg Confidence', 'Accuracy'],
            'Value': [
                len(predictions),
                f"{np.sum(predictions == 1)} ({np.sum(predictions == 1)/len(predictions)*100:.1f}%)",
                f"{escalation_count} ({escalation_count/len(predictions)*100:.1f}%)",
                f"{np.mean(max_probs):.3f}",
                f"{session_data.get('accuracy', 'N/A')}"
            ]
        }
        
        fig.add_trace(
            go.Table(
                header=dict(values=list(summary_data.keys()), fill_color='paleturquoise', align='left'),
                cells=dict(values=list(summary_data.values()), fill_color='lavender', align='left')
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=False,
            title_text="Stress Detection Session Report",
            title_font_size=24
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        
        return fig
    
    def print_escalation_alert(self, stress_prob, explanation, recommendation):
        """
        Print formatted escalation alert
        
        Args:
            stress_prob: Stress probability
            explanation: XAI explanation
            recommendation: GenAI recommendation
        """
        escalation_level, message = self.check_escalation(stress_prob)
        
        print("\n" + "="*80)
        print(message)
        print("="*80)
        print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Stress Probability: {stress_prob:.2%}")
        print(f"Escalation Level: {escalation_level}")
        print("\n" + "-"*80)
        print("EXPLANATION:")
        print("-"*80)
        print(explanation)
        print("\n" + "-"*80)
        print("RECOMMENDATIONS:")
        print("-"*80)
        print(recommendation)
        print("="*80 + "\n")
    
    def export_escalation_log(self, filepath):
        """Export escalation log to CSV"""
        if not self.escalation_log:
            print("No escalation events logged")
            return
        
        df = pd.DataFrame(self.escalation_log)
        df.to_csv(filepath, index=False)
        print(f"Escalation log exported to {filepath}")
    
    def plot_feature_importance(self, feature_names, importances, top_n=20, save_path=None):
        """
        Plot feature importance
        
        Args:
            feature_names: List of feature names
            importances: Feature importance values
            top_n: Number of top features to show
            save_path: Path to save plot
        """
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        # Create plot
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=top_importances,
            y=top_features,
            orientation='h',
            marker=dict(
                color=top_importances,
                colorscale='Viridis',
                showscale=True
            )
        ))
        
        fig.update_layout(
            title='Top Feature Importances for Stress Detection',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()


if __name__ == "__main__":
    # Example usage
    viz = StressVisualization(stress_threshold=0.7, escalation_threshold=0.85)
    
    # Simulate predictions
    np.random.seed(42)
    n_samples = 100
    y_pred_proba = np.random.rand(n_samples, 3)
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
    
    # Plot distributions
    viz.plot_stress_probability(y_pred_proba, save_path="../results/stress_probability.png")
    
    # Simulate real-time monitoring
    stress_history = y_pred_proba[:50, 1]
    viz.plot_realtime_monitor(stress_history, save_path="../results/realtime_monitor.html")
    
    # Check escalation
    for stress_prob in [0.5, 0.75, 0.9]:
        level, msg = viz.check_escalation(stress_prob)
        print(f"Stress: {stress_prob:.2f} -> {level}: {msg}")
