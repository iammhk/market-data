"""
📊 ADVANCED REAL-TIME MONITORING & ALERTS
Enhanced monitoring system with performance tracking and alerts for market predictions

This module provides comprehensive monitoring capabilities for real-time predictions
including alert systems, performance tracking, and visualization tools.
"""

import json
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any


class PredictionMonitor:
    """Advanced monitoring system for real-time predictions"""
    
    def __init__(self):
        self.predictions_log = []
        self.performance_metrics = {}
        self.alerts_enabled = True
        self.alert_thresholds = {
            'high_confidence': 1.0,  # Prediction difference > 1%
            'extreme_pcr': {'high': 1.5, 'low': 0.5},  # PCR thresholds
            'high_volume': 1000000,  # Volume threshold
            'price_gap': 0.8  # Price gap threshold
        }
        
    def log_prediction(self, prediction_result: Dict[str, Any]) -> None:
        """Log a prediction result with timestamp"""
        if prediction_result and prediction_result.get('success'):
            log_entry = {
                'timestamp': datetime.now(),
                'current_spot': prediction_result['current_spot'],
                'predicted_spot': prediction_result['predicted_spot'],
                'difference': prediction_result['difference'],
                'difference_pct': prediction_result['difference_pct'],
                'features': prediction_result['features_used'],
                'accuracy': None  # Will be filled when actual price is known
            }
            
            self.predictions_log.append(log_entry)
            
            # Check for alerts
            if self.alerts_enabled:
                self._check_alerts(log_entry)
            
            print(f"📝 Prediction logged at {log_entry['timestamp'].strftime('%H:%M:%S')}")
    
    def _check_alerts(self, log_entry: Dict[str, Any]) -> None:
        """Check for alert conditions and notify"""
        alerts = []
        
        # High confidence prediction alert
        if abs(log_entry['difference_pct']) > self.alert_thresholds['high_confidence']:
            direction = "BULLISH" if log_entry['difference'] > 0 else "BEARISH"
            alerts.append(f"🚨 HIGH CONFIDENCE {direction} SIGNAL: {log_entry['difference_pct']:+.2f}%")
        
        # Extreme PCR alert
        pcr = log_entry['features'].get('pcr_volume', 1)
        if pcr > self.alert_thresholds['extreme_pcr']['high']:
            alerts.append(f"📉 EXTREME BEARISH PCR: {pcr:.3f}")
        elif pcr < self.alert_thresholds['extreme_pcr']['low']:
            alerts.append(f"📈 EXTREME BULLISH PCR: {pcr:.3f}")
        
        # High volume alert
        total_volume = log_entry['features'].get('total_volume', 0)
        if total_volume > self.alert_thresholds['high_volume']:
            alerts.append(f"📊 HIGH VOLUME ACTIVITY: {total_volume:,.0f}")
        
        # Price gap alert
        price_change = log_entry['features'].get('price_change_pct', 0)
        if abs(price_change) > self.alert_thresholds['price_gap']:
            alerts.append(f"📈 SIGNIFICANT PRICE GAP: {price_change:+.2f}%")
        
        # Display alerts
        if alerts:
            print(f"\n🚨 ALERTS TRIGGERED")
            print("-" * 20)
            for alert in alerts:
                print(f"   {alert}")
            print()
    
    def get_recent_performance(self, minutes: int = 60) -> Dict[str, Any]:
        """Get performance metrics for recent predictions"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_predictions = [p for p in self.predictions_log if p['timestamp'] >= cutoff_time]
        
        if not recent_predictions:
            return {"message": "No recent predictions found"}
        
        # Calculate metrics
        differences = [p['difference_pct'] for p in recent_predictions]
        pcrs = [p['features'].get('pcr_volume', 1) for p in recent_predictions]
        volumes = [p['features'].get('total_volume', 0) for p in recent_predictions]
        
        metrics = {
            'count': len(recent_predictions),
            'avg_prediction_pct': np.mean(differences),
            'prediction_volatility': np.std(differences),
            'avg_pcr': np.mean(pcrs),
            'avg_volume': np.mean(volumes),
            'bullish_predictions': len([d for d in differences if d > 0]),
            'bearish_predictions': len([d for d in differences if d < 0]),
            'time_range': f"{recent_predictions[0]['timestamp'].strftime('%H:%M')} - {recent_predictions[-1]['timestamp'].strftime('%H:%M')}"
        }
        
        return metrics
    
    def plot_prediction_trend(self, hours: int = 2) -> None:
        """Plot prediction trend over time"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_predictions = [p for p in self.predictions_log if p['timestamp'] >= cutoff_time]
        
        if len(recent_predictions) < 2:
            print("❌ Insufficient data for trend plot")
            return
        
        # Create the plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        timestamps = [p['timestamp'] for p in recent_predictions]
        current_prices = [p['current_spot'] for p in recent_predictions]
        predicted_prices = [p['predicted_spot'] for p in recent_predictions]
        differences = [p['difference_pct'] for p in recent_predictions]
        pcrs = [p['features'].get('pcr_volume', 1) for p in recent_predictions]
        
        # Plot 1: Price comparison
        ax1.plot(timestamps, current_prices, 'b-', label='Current Spot', marker='o', markersize=4)
        ax1.plot(timestamps, predicted_prices, 'r--', label='Predicted Spot', marker='s', markersize=4)
        ax1.set_title('Spot Price vs Predictions')
        ax1.set_ylabel('Price (₹)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Prediction differences
        colors = ['green' if d > 0 else 'red' for d in differences]
        ax2.bar(timestamps, differences, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Prediction Differences')
        ax2.set_ylabel('Difference (%)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: PCR trend
        ax3.plot(timestamps, pcrs, 'purple', marker='o', markersize=4)
        ax3.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Neutral PCR')
        ax3.axhline(y=1.2, color='red', linestyle='--', alpha=0.5, label='Bearish PCR')
        ax3.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Bullish PCR')
        ax3.set_title('Put-Call Ratio Trend')
        ax3.set_ylabel('PCR')
        ax3.set_xlabel('Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show()
    
    def export_predictions(self, filename: Optional[str] = None) -> str:
        """Export predictions to CSV file"""
        if not self.predictions_log:
            print("❌ No predictions to export")
            return ""
        
        if filename is None:
            filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Convert to DataFrame
        export_data = []
        for pred in self.predictions_log:
            row = {
                'timestamp': pred['timestamp'],
                'current_spot': pred['current_spot'],
                'predicted_spot': pred['predicted_spot'],
                'difference': pred['difference'],
                'difference_pct': pred['difference_pct'],
                'pcr_volume': pred['features'].get('pcr_volume', 0),
                'pcr_oi': pred['features'].get('pcr_oi', 0),
                'total_volume': pred['features'].get('total_volume', 0),
                'total_oi': pred['features'].get('total_oi', 0),
                'call_atm_ltp': pred['features'].get('call_atm_ltp', 0),
                'put_atm_ltp': pred['features'].get('put_atm_ltp', 0)
            }
            export_data.append(row)
        
        df = pd.DataFrame(export_data)
        
        # Save to project data directory
        try:
            # Try to use project_root if available globally
            import sys
            if any('project_root' in frame.f_globals for frame in sys._current_frames().values()):
                # Find project_root from calling frame
                for frame in sys._current_frames().values():
                    if 'project_root' in frame.f_globals:
                        project_root = frame.f_globals['project_root']
                        export_path = os.path.join(project_root, 'data', filename)
                        break
                else:
                    export_path = filename
            else:
                # Default to current directory or create data directory
                if not os.path.exists('data'):
                    os.makedirs('data', exist_ok=True)
                export_path = os.path.join('data', filename)
        except:
            export_path = filename
        
        df.to_csv(export_path, index=False)
        print(f"✅ Predictions exported to: {export_path}")
        print(f"📊 Exported {len(df)} predictions")
        
        return export_path
    
    def display_dashboard(self) -> None:
        """Display a comprehensive monitoring dashboard"""
        print(f"\n📊 PREDICTION MONITORING DASHBOARD")
        print("=" * 45)
        print(f"🕐 Last Updated: {datetime.now().strftime('%H:%M:%S')}")
        
        if not self.predictions_log:
            print("❌ No predictions logged yet")
            return
        
        # Recent activity
        recent_metrics = self.get_recent_performance(60)
        if 'count' in recent_metrics:
            print(f"\n📈 LAST HOUR ACTIVITY")
            print("-" * 22)
            print(f"   🔮 Predictions: {recent_metrics['count']}")
            print(f"   📊 Avg Prediction: {recent_metrics['avg_prediction_pct']:+.2f}%")
            print(f"   📉 Volatility: {recent_metrics['prediction_volatility']:.2f}%")
            print(f"   🟢 Bullish: {recent_metrics['bullish_predictions']}")
            print(f"   🔴 Bearish: {recent_metrics['bearish_predictions']}")
            print(f"   🔄 Avg PCR: {recent_metrics['avg_pcr']:.3f}")
            print(f"   📊 Avg Volume: {recent_metrics['avg_volume']:,.0f}")
        
        # Latest prediction
        latest = self.predictions_log[-1]
        print(f"\n🔮 LATEST PREDICTION")
        print("-" * 20)
        print(f"   🕐 Time: {latest['timestamp'].strftime('%H:%M:%S')}")
        print(f"   💰 Current: ₹{latest['current_spot']:,.2f}")
        print(f"   🎯 Predicted: ₹{latest['predicted_spot']:,.2f}")
        print(f"   📈 Difference: {latest['difference_pct']:+.2f}%")
        
        # Alert status
        print(f"\n🚨 ALERT SETTINGS")
        print("-" * 16)
        print(f"   Status: {'🟢 Enabled' if self.alerts_enabled else '🔴 Disabled'}")
        print(f"   High Confidence: >{self.alert_thresholds['high_confidence']:.1f}%")
        print(f"   Extreme PCR: <{self.alert_thresholds['extreme_pcr']['low']:.1f} or >{self.alert_thresholds['extreme_pcr']['high']:.1f}")
        print(f"   High Volume: >{self.alert_thresholds['high_volume']:,.0f}")


def enhanced_live_prediction(make_live_prediction) -> Optional[Dict[str, Any]]:
    """
    Enhanced prediction function with monitoring
    
    Args:
        make_live_prediction: The prediction function to wrap with monitoring
        
    Returns:
        Prediction result dictionary or None
    """
    result = make_live_prediction(use_previous_day_data=True)
    
    if result and result.get('success'):
        # Additional analysis
        features = result['features_used']
        
        print(f"\n🔍 ENHANCED ANALYSIS")
        print("-" * 20)
        
        # Market sentiment analysis
        pcr = features.get('pcr_volume', 1)
        call_vol = features.get('call_total_volume', 0)
        put_vol = features.get('put_total_volume', 0)
        
        if pcr > 1.2:
            sentiment = "📉 Bearish (Fear dominant)"
        elif pcr < 0.8:
            sentiment = "📈 Bullish (Greed dominant)"
        else:
            sentiment = "⚖️ Neutral (Balanced)"
        
        print(f"🎭 Market Sentiment: {sentiment}")
        print(f"📊 Call Volume: {call_vol:,.0f}")
        print(f"📊 Put Volume: {put_vol:,.0f}")
        
        # Volume analysis
        total_vol = features.get('total_volume', 0)
        if total_vol > 1000000:
            vol_signal = "🔥 High activity"
        elif total_vol > 500000:
            vol_signal = "📊 Moderate activity"
        else:
            vol_signal = "😴 Low activity"
        
        print(f"📈 Volume Signal: {vol_signal}")
        
        # Price momentum
        price_change = features.get('price_change_pct', 0)
        if abs(price_change) > 0.5:
            momentum = f"🚀 Strong momentum ({price_change:+.2f}%)"
        else:
            momentum = f"🐌 Weak momentum ({price_change:+.2f}%)"
        
        print(f"⚡ Momentum: {momentum}")
        
    return result


def create_monitor_functions(monitor_instance: PredictionMonitor, make_live_prediction_func) -> Dict[str, callable]:
    """
    Create monitoring functions bound to a monitor instance
    
    Args:
        monitor_instance: Instance of PredictionMonitor
        make_live_prediction_func: The live prediction function to wrap
        
    Returns:
        Dictionary of monitoring functions
    """
    
    def start_monitoring():
        """Start the monitoring dashboard"""
        monitor_instance.display_dashboard()

    def quick_prediction():
        """Quick prediction with monitoring"""
        result = enhanced_live_prediction(make_live_prediction_func)
        if result:
            monitor_instance.log_prediction(result)
        return result

    def show_trends():
        """Show prediction trends"""
        monitor_instance.plot_prediction_trend()

    def export_data():
        """Export prediction data"""
        return monitor_instance.export_predictions()
    
    def enhanced_prediction():
        """Enhanced prediction with full monitoring"""
        result = enhanced_live_prediction(make_live_prediction_func)
        if result:
            monitor_instance.log_prediction(result) 
        return result
    
    return {
        'start_monitoring': start_monitoring,
        'quick_prediction': quick_prediction,
        'show_trends': show_trends,
        'export_data': export_data,
        'enhanced_live_prediction': enhanced_prediction
    }


def initialize_monitoring_system() -> tuple:
    """
    Initialize the monitoring system
    
    Returns:
        Tuple of (monitor_instance, function_dict)
    """
    print("📊 ADVANCED REAL-TIME MONITORING SYSTEM")
    print("=" * 50)
    
    monitor = PredictionMonitor()
    
    print(f"✅ ADVANCED MONITORING SYSTEM READY")
    print("-" * 40)
    print("📊 Available functions:")
    print("   • enhanced_live_prediction() - Prediction with monitoring")
    print("   • start_monitoring() - Show dashboard")
    print("   • quick_prediction() - Quick monitored prediction")
    print("   • show_trends() - Plot trends")
    print("   • export_data() - Export predictions to CSV")
    print(f"\n🎯 Monitor object available for advanced usage")
    
    return monitor


# Export main classes and functions
__all__ = [
    'PredictionMonitor',
    'enhanced_live_prediction',
    'create_monitor_functions',
    'initialize_monitoring_system'
]
