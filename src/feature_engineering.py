"""
Feature Engineering Module for AI Lead Scoring Engine
Handles real-time feature computation and data preprocessing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class LeadData:
    """Data structure for lead information"""
    lead_id: str
    company_name: str
    industry: str
    job_title: str
    company_size: Optional[int]
    lead_source: str
    created_date: datetime
    email_opens: int = 0
    website_visits: int = 0
    form_submissions: int = 0
    meeting_scheduled: bool = False
    meeting_attended: bool = False
    response_time_hours: Optional[float] = None
    location: Optional[str] = None
    funding_stage: Optional[str] = None


class FeatureEngineer:
    """Main feature engineering class for lead scoring"""
    
    def __init__(self, reference_data: pd.DataFrame = None):
        self.reference_data = reference_data
        self.feature_stats = {}
        self.industry_conversion_rates = {}
        self._initialize_reference_stats()
    
    def _initialize_reference_stats(self):
        """Initialize reference statistics for normalization"""
        if self.reference_data is not None:
            # Calculate industry conversion rates
            self.industry_conversion_rates = (
                self.reference_data.groupby('industry')['converted']
                .mean()
                .to_dict()
            )
            
            # Calculate feature statistics for normalization
            numeric_features = ['email_opens', 'website_visits', 'form_submissions']
            for feature in numeric_features:
                if feature in self.reference_data.columns:
                    self.feature_stats[feature] = {
                        'mean': self.reference_data[feature].mean(),
                        'std': self.reference_data[feature].std()
                    }
    
    def engineer_features(self, lead_data: LeadData) -> Dict[str, float]:
        """
        Main feature engineering function
        Returns dictionary of engineered features
        """
        features = {}
        
        # Behavioral features
        features.update(self._compute_behavioral_features(lead_data))
        
        # Demographic features
        features.update(self._compute_demographic_features(lead_data))
        
        # Temporal features
        features.update(self._compute_temporal_features(lead_data))
        
        # Interaction quality features
        features.update(self._compute_interaction_features(lead_data))
        
        # Normalize features
        features = self._normalize_features(features)
        
        return features
    
    def _compute_behavioral_features(self, lead_data: LeadData) -> Dict[str, float]:
        """Compute behavioral engagement features"""
        features = {}
        
        # Engagement velocity
        days_since_creation = (datetime.now() - lead_data.created_date).days
        if days_since_creation > 0:
            engagement_score = (
                lead_data.email_opens * 0.3 +
                lead_data.website_visits * 0.4 +
                lead_data.form_submissions * 0.3
            ) / days_since_creation
        else:
            engagement_score = 0.0
        
        features['engagement_velocity'] = min(engagement_score, 10.0)  # Cap at 10
        
        # Response latency (inverse relationship - faster is better)
        if lead_data.response_time_hours is not None:
            features['response_latency_score'] = max(0, 1 - (lead_data.response_time_hours / 24))
        else:
            features['response_latency_score'] = 0.0
        
        # Content consumption intensity
        total_interactions = lead_data.email_opens + lead_data.website_visits + lead_data.form_submissions
        features['content_consumption_intensity'] = min(total_interactions / 10, 1.0)
        
        return features
    
    def _compute_demographic_features(self, lead_data: LeadData) -> Dict[str, float]:
        """Compute demographic and company-related features"""
        features = {}
        
        # Decision maker probability based on job title
        decision_maker_keywords = [
            'ceo', 'cto', 'cfo', 'vp', 'director', 'head', 'manager', 'founder'
        ]
        title_lower = lead_data.job_title.lower()
        decision_maker_score = sum(1 for keyword in decision_maker_keywords if keyword in title_lower)
        features['decision_maker_probability'] = min(decision_maker_score / 3, 1.0)
        
        # Company maturity score
        if lead_data.company_size:
            if lead_data.company_size < 10:
                maturity_score = 0.2
            elif lead_data.company_size < 50:
                maturity_score = 0.4
            elif lead_data.company_size < 200:
                maturity_score = 0.6
            elif lead_data.company_size < 1000:
                maturity_score = 0.8
            else:
                maturity_score = 1.0
        else:
            maturity_score = 0.5  # Default for unknown
        
        features['company_maturity_score'] = maturity_score
        
        # Industry affinity
        if lead_data.industry in self.industry_conversion_rates:
            features['industry_affinity'] = self.industry_conversion_rates[lead_data.industry]
        else:
            features['industry_affinity'] = 0.1  # Default for unknown industry
        
        # Funding stage impact
        funding_scores = {
            'seed': 0.3,
            'series_a': 0.5,
            'series_b': 0.7,
            'series_c': 0.8,
            'public': 0.9
        }
        features['funding_stage_score'] = funding_scores.get(lead_data.funding_stage, 0.5)
        
        return features
    
    def _compute_temporal_features(self, lead_data: LeadData) -> Dict[str, float]:
        """Compute time-based features"""
        features = {}
        
        # Quarter end proximity
        now = datetime.now()
        quarter_end = datetime(now.year, ((now.month - 1) // 3 + 1) * 3, 1)
        if quarter_end.month == 3:
            quarter_end = quarter_end.replace(month=3, day=31)
        elif quarter_end.month == 6:
            quarter_end = quarter_end.replace(month=6, day=30)
        elif quarter_end.month == 9:
            quarter_end = quarter_end.replace(month=9, day=30)
        else:
            quarter_end = quarter_end.replace(month=12, day=31)
        
        days_to_quarter_end = (quarter_end - now).days
        features['quarter_end_proximity'] = max(0, 1 - (days_to_quarter_end / 90))
        
        # Seasonal intent
        month = now.month
        seasonal_multipliers = {
            1: 0.8, 2: 0.9, 3: 1.2, 4: 1.0, 5: 1.0, 6: 1.1,
            7: 0.9, 8: 0.8, 9: 1.3, 10: 1.1, 11: 1.0, 12: 1.2
        }
        features['seasonal_intent'] = seasonal_multipliers.get(month, 1.0)
        
        # Lead age decay
        days_since_creation = (now - lead_data.created_date).days
        features['lead_age_decay'] = np.exp(-days_since_creation / 30)
        
        # Time of day preference (assuming peak hours 9-11 AM and 2-4 PM)
        hour = now.hour
        if 9 <= hour <= 11 or 14 <= hour <= 16:
            features['peak_hour_activity'] = 1.0
        else:
            features['peak_hour_activity'] = 0.5
        
        return features
    
    def _compute_interaction_features(self, lead_data: LeadData) -> Dict[str, float]:
        """Compute interaction quality features"""
        features = {}
        
        # Meeting attendance rate
        if lead_data.meeting_scheduled:
            features['meeting_attendance_rate'] = 1.0 if lead_data.meeting_attended else 0.0
        else:
            features['meeting_attendance_rate'] = 0.5  # Neutral for no meetings
        
        # Engagement consistency
        total_activities = lead_data.email_opens + lead_data.website_visits + lead_data.form_submissions
        if total_activities > 0:
            # Calculate coefficient of variation for engagement consistency
            activities = [lead_data.email_opens, lead_data.website_visits, lead_data.form_submissions]
            if np.std(activities) > 0:
                cv = np.std(activities) / np.mean(activities)
                features['engagement_consistency'] = max(0, 1 - cv)
            else:
                features['engagement_consistency'] = 1.0
        else:
            features['engagement_consistency'] = 0.0
        
        # Lead source quality
        high_quality_sources = ['referral', 'partner', 'event', 'webinar']
        features['source_quality'] = 1.0 if lead_data.lead_source in high_quality_sources else 0.5
        
        return features
    
    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalize features to 0-1 range and handle outliers"""
        normalized_features = {}
        
        for feature_name, value in features.items():
            # Clip outliers
            clipped_value = np.clip(value, 0, 1)
            
            # Apply sigmoid normalization for some features
            if feature_name in ['engagement_velocity', 'content_consumption_intensity']:
                normalized_value = 1 / (1 + np.exp(-5 * (clipped_value - 0.5)))
            else:
                normalized_value = clipped_value
            
            normalized_features[feature_name] = float(normalized_value)
        
        return normalized_features
    
    def get_feature_importance_weights(self) -> Dict[str, float]:
        """Return feature importance weights for model training"""
        return {
            'engagement_velocity': 0.25,
            'decision_maker_probability': 0.20,
            'industry_affinity': 0.15,
            'quarter_end_proximity': 0.10,
            'response_latency_score': 0.10,
            'company_maturity_score': 0.08,
            'meeting_attendance_rate': 0.05,
            'seasonal_intent': 0.03,
            'engagement_consistency': 0.02,
            'source_quality': 0.02
        }


class FeatureValidator:
    """Validates feature quality and detects anomalies"""
    
    def __init__(self, reference_stats: Dict = None):
        self.reference_stats = reference_stats or {}
        self.anomaly_threshold = 3.0  # Standard deviations
    
    def validate_features(self, features: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Validate feature values and return issues"""
        issues = []
        
        for feature_name, value in features.items():
            # Check for NaN or infinite values
            if not np.isfinite(value):
                issues.append(f"Invalid value for {feature_name}: {value}")
                continue
            
            # Check for out-of-range values
            if value < 0 or value > 1:
                issues.append(f"Out of range value for {feature_name}: {value}")
                continue
            
            # Check for statistical anomalies if reference stats available
            if feature_name in self.reference_stats:
                ref_mean = self.reference_stats[feature_name]['mean']
                ref_std = self.reference_stats[feature_name]['std']
                
                if ref_std > 0:
                    z_score = abs(value - ref_mean) / ref_std
                    if z_score > self.anomaly_threshold:
                        issues.append(f"Anomalous value for {feature_name}: z-score {z_score:.2f}")
        
        return len(issues) == 0, issues


# Example usage and testing
if __name__ == "__main__":
    # Sample lead data
    sample_lead = LeadData(
        lead_id="L12345",
        company_name="TechCorp Inc",
        industry="Technology",
        job_title="VP of Engineering",
        company_size=150,
        lead_source="website",
        created_date=datetime.now() - timedelta(days=5),
        email_opens=3,
        website_visits=8,
        form_submissions=1,
        meeting_scheduled=True,
        meeting_attended=True,
        response_time_hours=2.5,
        funding_stage="series_b"
    )
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Engineer features
    features = feature_engineer.engineer_features(sample_lead)
    
    print("Engineered Features:")
    for feature, value in features.items():
        print(f"{feature}: {value:.4f}")
    
    # Validate features
    validator = FeatureValidator()
    is_valid, issues = validator.validate_features(features)
    
    if not is_valid:
        print("\nValidation Issues:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("\nAll features are valid!") 