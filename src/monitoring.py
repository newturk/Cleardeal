"""
Monitoring and Drift Detection for AI Lead Scoring Engine
Handles data drift, model drift, and performance monitoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import json
import redis
from scipy import stats
from dataclasses import dataclass
import asyncio
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Data structure for drift alerts"""
    alert_id: str
    alert_type: str  # 'data_drift', 'model_drift', 'performance_degradation'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: datetime
    metrics: Dict[str, float]
    threshold: float
    current_value: float


class DataDriftMonitor:
    """Monitors data distribution changes"""
    
    def __init__(self, reference_data: pd.DataFrame, drift_threshold: float = 0.05):
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.feature_stats = {}
        self.alert_history = deque(maxlen=1000)
        self._initialize_reference_stats()
    
    def _initialize_reference_stats(self):
        """Initialize reference statistics for each feature"""
        numeric_features = self.reference_data.select_dtypes(include=[np.number]).columns
        
        for feature in numeric_features:
            if feature != 'converted':  # Skip target variable
                feature_data = self.reference_data[feature].dropna()
                if len(feature_data) > 0:
                    self.feature_stats[feature] = {
                        'mean': feature_data.mean(),
                        'std': feature_data.std(),
                        'distribution': feature_data.values
                    }
    
    def detect_feature_drift(self, current_data: pd.DataFrame) -> List[DriftAlert]:
        """Detect drift in individual features"""
        alerts = []
        
        for feature, ref_stats in self.feature_stats.items():
            if feature in current_data.columns:
                current_feature = current_data[feature].dropna()
                
                if len(current_feature) > 0:
                    # Kolmogorov-Smirnov test for distribution changes
                    ks_statistic, p_value = stats.ks_2samp(
                        ref_stats['distribution'], 
                        current_feature.values
                    )
                    
                    if p_value < self.drift_threshold:
                        severity = self._determine_severity(p_value)
                        
                        alert = DriftAlert(
                            alert_id=f"drift_{feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            alert_type="data_drift",
                            severity=severity,
                            message=f"Feature drift detected in {feature}: p-value={p_value:.4f}",
                            timestamp=datetime.now(),
                            metrics={'ks_statistic': ks_statistic, 'p_value': p_value},
                            threshold=self.drift_threshold,
                            current_value=p_value
                        )
                        
                        alerts.append(alert)
                        self.alert_history.append(alert)
        
        return alerts
    
    def detect_covariate_shift(self, current_data: pd.DataFrame) -> Optional[DriftAlert]:
        """Detect overall covariate shift using Population Stability Index"""
        try:
            # Calculate PSI for each feature
            psi_scores = {}
            total_psi = 0
            feature_count = 0
            
            for feature, ref_stats in self.feature_stats.items():
                if feature in current_data.columns:
                    psi = self._calculate_psi(ref_stats['distribution'], current_data[feature].dropna())
                    psi_scores[feature] = psi
                    total_psi += psi
                    feature_count += 1
            
            if feature_count > 0:
                avg_psi = total_psi / feature_count
                
                if avg_psi > 0.25:  # Significant shift threshold
                    severity = self._determine_severity(1 - avg_psi)
                    
                    alert = DriftAlert(
                        alert_id=f"covariate_shift_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        alert_type="data_drift",
                        severity=severity,
                        message=f"Covariate shift detected: average PSI={avg_psi:.4f}",
                        timestamp=datetime.now(),
                        metrics={'avg_psi': avg_psi, 'feature_psi': psi_scores},
                        threshold=0.25,
                        current_value=avg_psi
                    )
                    
                    self.alert_history.append(alert)
                    return alert
            
            return None
            
        except Exception as e:
            logger.error(f"Error calculating covariate shift: {e}")
            return None
    
    def _calculate_psi(self, reference_data: np.ndarray, current_data: np.ndarray) -> float:
        """Calculate Population Stability Index"""
        try:
            # Create bins for both distributions
            all_data = np.concatenate([reference_data, current_data])
            bins = np.histogram_bin_edges(all_data, bins=10)
            
            # Calculate histograms
            ref_hist, _ = np.histogram(reference_data, bins=bins)
            curr_hist, _ = np.histogram(current_data, bins=bins)
            
            # Normalize to probabilities
            ref_prob = ref_hist / np.sum(ref_hist)
            curr_prob = curr_hist / np.sum(curr_hist)
            
            # Calculate PSI
            psi = 0
            for i in range(len(ref_prob)):
                if ref_prob[i] > 0 and curr_prob[i] > 0:
                    psi += (curr_prob[i] - ref_prob[i]) * np.log(curr_prob[i] / ref_prob[i])
            
            return psi
            
        except Exception as e:
            logger.error(f"Error calculating PSI: {e}")
            return 0.0
    
    def _determine_severity(self, p_value: float) -> str:
        """Determine alert severity based on p-value"""
        if p_value < 0.001:
            return "critical"
        elif p_value < 0.01:
            return "high"
        elif p_value < 0.05:
            return "medium"
        else:
            return "low"


class ModelPerformanceMonitor:
    """Monitors model performance degradation"""
    
    def __init__(self, baseline_metrics: Dict[str, float], degradation_threshold: float = 0.05):
        self.baseline_metrics = baseline_metrics
        self.degradation_threshold = degradation_threshold
        self.performance_history = deque(maxlen=1000)
        self.alert_history = deque(maxlen=1000)
    
    def monitor_performance(self, recent_predictions: List[float], 
                          actual_outcomes: List[int]) -> List[DriftAlert]:
        """Monitor model performance and detect degradation"""
        alerts = []
        
        if len(recent_predictions) < 50:  # Need minimum sample size
            return alerts
        
        # Calculate current metrics
        current_metrics = self._calculate_metrics(recent_predictions, actual_outcomes)
        
        # Store performance history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'metrics': current_metrics,
            'sample_size': len(recent_predictions)
        })
        
        # Check for degradation in each metric
        for metric_name, baseline_value in self.baseline_metrics.items():
            if metric_name in current_metrics:
                current_value = current_metrics[metric_name]
                degradation = baseline_value - current_value
                
                if degradation > self.degradation_threshold:
                    severity = self._determine_degradation_severity(degradation)
                    
                    alert = DriftAlert(
                        alert_id=f"performance_{metric_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        alert_type="model_drift",
                        severity=severity,
                        message=f"Performance degradation in {metric_name}: {degradation:.4f}",
                        timestamp=datetime.now(),
                        metrics={metric_name: current_value},
                        threshold=self.degradation_threshold,
                        current_value=degradation
                    )
                    
                    alerts.append(alert)
                    self.alert_history.append(alert)
        
        return alerts
    
    def _calculate_metrics(self, predictions: List[float], outcomes: List[int]) -> Dict[str, float]:
        """Calculate performance metrics"""
        try:
            predictions = np.array(predictions)
            outcomes = np.array(outcomes)
            
            # Basic metrics
            auc = stats.roc_auc_score(outcomes, predictions)
            
            # Precision and recall at different thresholds
            thresholds = [0.3, 0.5, 0.7]
            metrics = {'auc': auc}
            
            for threshold in thresholds:
                pred_labels = (predictions >= threshold).astype(int)
                precision = stats.precision_score(outcomes, pred_labels, zero_division=0)
                recall = stats.recall_score(outcomes, pred_labels, zero_division=0)
                
                metrics[f'precision_{threshold}'] = precision
                metrics[f'recall_{threshold}'] = recall
            
            # Precision at top 20%
            top_20_idx = np.argsort(predictions)[-int(len(predictions) * 0.2):]
            precision_top_20 = stats.precision_score(outcomes[top_20_idx], 
                                                   (predictions[top_20_idx] >= 0.5).astype(int),
                                                   zero_division=0)
            metrics['precision_top_20'] = precision_top_20
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def _determine_degradation_severity(self, degradation: float) -> str:
        """Determine alert severity based on degradation amount"""
        if degradation > 0.15:
            return "critical"
        elif degradation > 0.10:
            return "high"
        elif degradation > 0.05:
            return "medium"
        else:
            return "low"


class MonitoringDashboard:
    """Comprehensive monitoring dashboard"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.data_drift_monitor = None
        self.performance_monitor = None
        self.monitoring_active = False
    
    def initialize_monitoring(self, reference_data: pd.DataFrame, baseline_metrics: Dict[str, float]):
        """Initialize monitoring components"""
        self.data_drift_monitor = DataDriftMonitor(reference_data)
        self.performance_monitor = ModelPerformanceMonitor(baseline_metrics)
        self.monitoring_active = True
        
        logger.info("Monitoring dashboard initialized")
    
    async def run_monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                await self._check_data_drift()
                await self._check_model_performance()
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _check_data_drift(self):
        """Check for data drift using recent data"""
        try:
            # Get recent data from Redis
            recent_data = self._get_recent_data()
            
            if recent_data is not None and len(recent_data) > 100:
                # Detect feature drift
                feature_alerts = self.data_drift_monitor.detect_feature_drift(recent_data)
                
                # Detect covariate shift
                covariate_alert = self.data_drift_monitor.detect_covariate_shift(recent_data)
                
                # Process alerts
                all_alerts = feature_alerts + ([covariate_alert] if covariate_alert else [])
                
                for alert in all_alerts:
                    await self._process_alert(alert)
                    
        except Exception as e:
            logger.error(f"Error checking data drift: {e}")
    
    async def _check_model_performance(self):
        """Check model performance using recent predictions"""
        try:
            # Get recent predictions and outcomes
            recent_predictions, recent_outcomes = self._get_recent_predictions()
            
            if recent_predictions and recent_outcomes:
                performance_alerts = self.performance_monitor.monitor_performance(
                    recent_predictions, recent_outcomes
                )
                
                for alert in performance_alerts:
                    await self._process_alert(alert)
                    
        except Exception as e:
            logger.error(f"Error checking model performance: {e}")
    
    def _get_recent_data(self) -> Optional[pd.DataFrame]:
        """Get recent data from Redis"""
        try:
            # Get recent lead data from Redis
            recent_leads = self.redis_client.lrange("recent_leads", 0, 999)
            
            if recent_leads:
                data_list = []
                for lead_json in recent_leads:
                    try:
                        lead_data = json.loads(lead_json)
                        data_list.append(lead_data)
                    except json.JSONDecodeError:
                        continue
                
                if data_list:
                    return pd.DataFrame(data_list)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting recent data: {e}")
            return None
    
    def _get_recent_predictions(self) -> Tuple[Optional[List[float]], Optional[List[int]]]:
        """Get recent predictions and outcomes from Redis"""
        try:
            # Get recent predictions from Redis
            prediction_logs = self.redis_client.lrange("prediction_logs", 0, 999)
            
            if prediction_logs:
                predictions = []
                outcomes = []
                
                for log_json in prediction_logs:
                    try:
                        log_data = json.loads(log_json)
                        predictions.append(log_data.get('score', 0.0))
                        # Note: outcomes would need to be tracked separately
                        # For now, we'll use a placeholder
                        outcomes.append(0)  # Placeholder
                    except json.JSONDecodeError:
                        continue
                
                return predictions, outcomes
            
            return None, None
            
        except Exception as e:
            logger.error(f"Error getting recent predictions: {e}")
            return None, None
    
    async def _process_alert(self, alert: DriftAlert):
        """Process and store alert"""
        try:
            # Store alert in Redis
            alert_data = {
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type,
                'severity': alert.severity,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'metrics': alert.metrics,
                'threshold': alert.threshold,
                'current_value': alert.current_value
            }
            
            self.redis_client.lpush("alerts", json.dumps(alert_data))
            self.redis_client.ltrim("alerts", 0, 999)  # Keep last 1000 alerts
            
            # Log alert
            logger.warning(f"Alert: {alert.message}")
            
            # Send notification for high severity alerts
            if alert.severity in ['high', 'critical']:
                await self._send_notification(alert)
                
        except Exception as e:
            logger.error(f"Error processing alert: {e}")
    
    async def _send_notification(self, alert: DriftAlert):
        """Send notification for high severity alerts"""
        # In production, this would send email/Slack notifications
        logger.critical(f"CRITICAL ALERT: {alert.message}")
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get comprehensive dashboard metrics"""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'monitoring_active': self.monitoring_active,
                'alerts': self._get_recent_alerts(),
                'performance_trends': self._get_performance_trends(),
                'data_quality': self._get_data_quality_metrics()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting dashboard metrics: {e}")
            return {'error': str(e)}
    
    def _get_recent_alerts(self) -> List[Dict]:
        """Get recent alerts from Redis"""
        try:
            alerts = self.redis_client.lrange("alerts", 0, 49)  # Last 50 alerts
            return [json.loads(alert) for alert in alerts if alert]
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
    
    def _get_performance_trends(self) -> Dict[str, List]:
        """Get performance trends over time"""
        if self.performance_monitor:
            history = list(self.performance_monitor.performance_history)
            if history:
                return {
                    'timestamps': [h['timestamp'].isoformat() for h in history],
                    'auc_scores': [h['metrics'].get('auc', 0) for h in history],
                    'precision_scores': [h['metrics'].get('precision_top_20', 0) for h in history]
                }
        return {}
    
    def _get_data_quality_metrics(self) -> Dict[str, float]:
        """Get data quality metrics"""
        try:
            recent_data = self._get_recent_data()
            if recent_data is not None:
                return {
                    'missing_data_rate': recent_data.isnull().mean().mean(),
                    'data_volume': len(recent_data),
                    'feature_count': len(recent_data.columns)
                }
        except Exception as e:
            logger.error(f"Error getting data quality metrics: {e}")
        
        return {}


# Example usage
if __name__ == "__main__":
    # Create sample reference data
    np.random.seed(42)
    reference_data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.normal(5, 2, 1000),
        'feature_3': np.random.exponential(1, 1000),
        'converted': np.random.choice([0, 1], 1000, p=[0.8, 0.2])
    })
    
    # Initialize monitoring
    baseline_metrics = {'auc': 0.85, 'precision_top_20': 0.70}
    
    # Create drift monitor
    drift_monitor = DataDriftMonitor(reference_data)
    
    # Create performance monitor
    performance_monitor = ModelPerformanceMonitor(baseline_metrics)
    
    print("Monitoring components initialized successfully!")
    print(f"Reference data shape: {reference_data.shape}")
    print(f"Baseline metrics: {baseline_metrics}") 