"""
Unit tests for feature engineering module
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.feature_engineering import (
    LeadData, 
    FeatureEngineer, 
    FeatureValidator
)


class TestLeadData:
    """Test LeadData dataclass"""
    
    def test_lead_data_creation(self):
        """Test creating LeadData object with all fields"""
        lead = LeadData(
            lead_id="TEST001",
            company_name="Test Corp",
            industry="Technology",
            job_title="VP Engineering",
            company_size=200,
            lead_source="website",
            created_date=datetime.now(),
            email_opens=5,
            website_visits=10,
            form_submissions=2,
            meeting_scheduled=True,
            meeting_attended=True,
            response_time_hours=2.5,
            location="San Francisco",
            funding_stage="series_b"
        )
        
        assert lead.lead_id == "TEST001"
        assert lead.company_name == "Test Corp"
        assert lead.industry == "Technology"
        assert lead.job_title == "VP Engineering"
        assert lead.company_size == 200
        assert lead.email_opens == 5
        assert lead.website_visits == 10
        assert lead.form_submissions == 2
        assert lead.meeting_scheduled is True
        assert lead.meeting_attended is True
        assert lead.response_time_hours == 2.5
        assert lead.location == "San Francisco"
        assert lead.funding_stage == "series_b"
    
    def test_lead_data_defaults(self):
        """Test LeadData object with default values"""
        lead = LeadData(
            lead_id="TEST002",
            company_name="Test Corp",
            industry="Technology",
            job_title="Engineer",
            company_size=None,
            lead_source="website",
            created_date=datetime.now()
        )
        
        assert lead.email_opens == 0
        assert lead.website_visits == 0
        assert lead.form_submissions == 0
        assert lead.meeting_scheduled is False
        assert lead.meeting_attended is False
        assert lead.response_time_hours is None
        assert lead.location is None
        assert lead.funding_stage is None


class TestFeatureEngineer:
    """Test FeatureEngineer class"""
    
    @pytest.fixture
    def sample_lead(self):
        """Create a sample lead for testing"""
        return LeadData(
            lead_id="TEST001",
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
    
    @pytest.fixture
    def reference_data(self):
        """Create sample reference data"""
        np.random.seed(42)
        data = pd.DataFrame({
            'industry': ['Technology', 'Finance', 'Healthcare'] * 100,
            'email_opens': np.random.poisson(3, 300),
            'website_visits': np.random.poisson(5, 300),
            'form_submissions': np.random.poisson(1, 300),
            'converted': np.random.choice([0, 1], 300, p=[0.8, 0.2])
        })
        return data
    
    def test_feature_engineer_initialization(self, reference_data):
        """Test FeatureEngineer initialization"""
        engineer = FeatureEngineer(reference_data)
        
        assert engineer.reference_data is not None
        assert 'Technology' in engineer.industry_conversion_rates
        assert 'email_opens' in engineer.feature_stats
    
    def test_behavioral_features(self, sample_lead):
        """Test behavioral feature computation"""
        engineer = FeatureEngineer()
        features = engineer._compute_behavioral_features(sample_lead)
        
        assert 'engagement_velocity' in features
        assert 'response_latency_score' in features
        assert 'content_consumption_intensity' in features
        
        # Test engagement velocity calculation
        expected_engagement = (3 * 0.3 + 8 * 0.4 + 1 * 0.3) / 5
        assert abs(features['engagement_velocity'] - expected_engagement) < 0.01
        
        # Test response latency score
        expected_latency_score = max(0, 1 - (2.5 / 24))
        assert abs(features['response_latency_score'] - expected_latency_score) < 0.01
        
        # Test content consumption intensity
        expected_intensity = min((3 + 8 + 1) / 10, 1.0)
        assert abs(features['content_consumption_intensity'] - expected_intensity) < 0.01
    
    def test_demographic_features(self, sample_lead):
        """Test demographic feature computation"""
        engineer = FeatureEngineer()
        features = engineer._compute_demographic_features(sample_lead)
        
        assert 'decision_maker_probability' in features
        assert 'company_maturity_score' in features
        assert 'industry_affinity' in features
        assert 'funding_stage_score' in features
        
        # Test decision maker probability
        assert features['decision_maker_probability'] > 0  # "VP" should be detected
        
        # Test company maturity score
        assert features['company_maturity_score'] == 0.6  # 150 employees
        
        # Test funding stage score
        assert features['funding_stage_score'] == 0.7  # series_b
    
    def test_temporal_features(self, sample_lead):
        """Test temporal feature computation"""
        engineer = FeatureEngineer()
        features = engineer._compute_temporal_features(sample_lead)
        
        assert 'quarter_end_proximity' in features
        assert 'seasonal_intent' in features
        assert 'lead_age_decay' in features
        assert 'peak_hour_activity' in features
        
        # Test lead age decay
        expected_decay = np.exp(-5 / 30)  # 5 days old
        assert abs(features['lead_age_decay'] - expected_decay) < 0.01
        
        # Test seasonal intent (should be a valid multiplier)
        assert 0.5 <= features['seasonal_intent'] <= 1.5
    
    def test_interaction_features(self, sample_lead):
        """Test interaction feature computation"""
        engineer = FeatureEngineer()
        features = engineer._compute_interaction_features(sample_lead)
        
        assert 'meeting_attendance_rate' in features
        assert 'engagement_consistency' in features
        assert 'source_quality' in features
        
        # Test meeting attendance rate
        assert features['meeting_attendance_rate'] == 1.0  # Scheduled and attended
        
        # Test source quality
        assert features['source_quality'] == 0.5  # website is not high quality
    
    def test_feature_normalization(self, sample_lead):
        """Test feature normalization"""
        engineer = FeatureEngineer()
        
        # Create features with extreme values
        raw_features = {
            'engagement_velocity': 15.0,  # Above cap
            'response_latency_score': -0.5,  # Below 0
            'content_consumption_intensity': 0.8,  # Normal value
            'decision_maker_probability': 1.5  # Above 1
        }
        
        normalized = engineer._normalize_features(raw_features)
        
        # Check that all values are in [0, 1] range
        for value in normalized.values():
            assert 0 <= value <= 1
        
        # Check that extreme values are clipped
        assert normalized['engagement_velocity'] <= 1.0
        assert normalized['response_latency_score'] >= 0.0
        assert normalized['decision_maker_probability'] <= 1.0
    
    def test_complete_feature_engineering(self, sample_lead):
        """Test complete feature engineering pipeline"""
        engineer = FeatureEngineer()
        features = engineer.engineer_features(sample_lead)
        
        # Check that all feature types are present
        expected_features = [
            'engagement_velocity', 'response_latency_score', 'content_consumption_intensity',
            'decision_maker_probability', 'company_maturity_score', 'industry_affinity',
            'funding_stage_score', 'quarter_end_proximity', 'seasonal_intent',
            'lead_age_decay', 'peak_hour_activity', 'meeting_attendance_rate',
            'engagement_consistency', 'source_quality'
        ]
        
        for feature in expected_features:
            assert feature in features
            assert 0 <= features[feature] <= 1  # All features should be normalized
    
    def test_feature_importance_weights(self):
        """Test feature importance weights"""
        engineer = FeatureEngineer()
        weights = engineer.get_feature_importance_weights()
        
        # Check that weights sum to 1.0
        assert abs(sum(weights.values()) - 1.0) < 0.01
        
        # Check that all weights are positive
        for weight in weights.values():
            assert weight > 0
        
        # Check that engagement_velocity has highest weight
        assert weights['engagement_velocity'] == max(weights.values())


class TestFeatureValidator:
    """Test FeatureValidator class"""
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features for testing"""
        return {
            'engagement_velocity': 0.75,
            'decision_maker_probability': 0.8,
            'industry_affinity': 0.6,
            'response_latency_score': 0.9
        }
    
    @pytest.fixture
    def reference_stats(self):
        """Create sample reference statistics"""
        return {
            'engagement_velocity': {'mean': 0.5, 'std': 0.2},
            'decision_maker_probability': {'mean': 0.3, 'std': 0.3},
            'industry_affinity': {'mean': 0.4, 'std': 0.1}
        }
    
    def test_validator_initialization(self):
        """Test FeatureValidator initialization"""
        validator = FeatureValidator()
        assert validator.anomaly_threshold == 3.0
        
        validator_custom = FeatureValidator(anomaly_threshold=2.0)
        assert validator_custom.anomaly_threshold == 2.0
    
    def test_validate_valid_features(self, sample_features):
        """Test validation of valid features"""
        validator = FeatureValidator()
        is_valid, issues = validator.validate_features(sample_features)
        
        assert is_valid is True
        assert len(issues) == 0
    
    def test_validate_invalid_values(self):
        """Test validation of invalid feature values"""
        validator = FeatureValidator()
        
        # Test NaN values
        features_with_nan = {
            'engagement_velocity': np.nan,
            'decision_maker_probability': 0.8
        }
        is_valid, issues = validator.validate_features(features_with_nan)
        assert is_valid is False
        assert len(issues) == 1
        assert "Invalid value" in issues[0]
        
        # Test infinite values
        features_with_inf = {
            'engagement_velocity': np.inf,
            'decision_maker_probability': 0.8
        }
        is_valid, issues = validator.validate_features(features_with_inf)
        assert is_valid is False
        assert len(issues) == 1
        assert "Invalid value" in issues[0]
        
        # Test out-of-range values
        features_out_of_range = {
            'engagement_velocity': 1.5,  # Above 1
            'decision_maker_probability': -0.1  # Below 0
        }
        is_valid, issues = validator.validate_features(features_out_of_range)
        assert is_valid is False
        assert len(issues) == 2
        assert all("Out of range" in issue for issue in issues)
    
    def test_validate_anomalous_values(self, reference_stats):
        """Test validation with anomalous values"""
        validator = FeatureValidator(reference_stats)
        
        # Test normal values
        normal_features = {
            'engagement_velocity': 0.5,  # Close to mean
            'decision_maker_probability': 0.3  # Close to mean
        }
        is_valid, issues = validator.validate_features(normal_features)
        assert is_valid is True
        assert len(issues) == 0
        
        # Test anomalous values
        anomalous_features = {
            'engagement_velocity': 1.5,  # 5 standard deviations above mean
            'decision_maker_probability': 0.3
        }
        is_valid, issues = validator.validate_features(anomalous_features)
        assert is_valid is False
        assert len(issues) == 1
        assert "Anomalous value" in issues[0]
    
    def test_validate_mixed_issues(self, reference_stats):
        """Test validation with multiple types of issues"""
        validator = FeatureValidator(reference_stats)
        
        features_with_issues = {
            'engagement_velocity': np.nan,  # Invalid value
            'decision_maker_probability': 1.5,  # Out of range
            'industry_affinity': 0.8,  # Anomalous (4 std devs above mean)
            'unknown_feature': 0.5  # Not in reference stats
        }
        
        is_valid, issues = validator.validate_features(features_with_issues)
        assert is_valid is False
        assert len(issues) == 3  # Should catch 3 issues


class TestIntegration:
    """Integration tests for feature engineering pipeline"""
    
    def test_end_to_end_pipeline(self):
        """Test complete feature engineering pipeline"""
        # Create sample lead
        lead = LeadData(
            lead_id="INTEGRATION_TEST",
            company_name="Integration Corp",
            industry="Technology",
            job_title="CTO",
            company_size=500,
            lead_source="referral",
            created_date=datetime.now() - timedelta(days=3),
            email_opens=10,
            website_visits=15,
            form_submissions=3,
            meeting_scheduled=True,
            meeting_attended=True,
            response_time_hours=1.0,
            funding_stage="series_c"
        )
        
        # Create feature engineer
        engineer = FeatureEngineer()
        
        # Engineer features
        features = engineer.engineer_features(lead)
        
        # Validate features
        validator = FeatureValidator()
        is_valid, issues = validator.validate_features(features)
        
        # Assertions
        assert is_valid is True
        assert len(issues) == 0
        assert len(features) > 0
        
        # Check specific feature expectations
        assert features['decision_maker_probability'] > 0.5  # CTO should be high
        assert features['company_maturity_score'] > 0.7  # 500 employees
        assert features['source_quality'] == 1.0  # referral is high quality
        assert features['meeting_attendance_rate'] == 1.0  # attended meeting
    
    def test_edge_cases(self):
        """Test edge cases in feature engineering"""
        # Test with minimal data
        minimal_lead = LeadData(
            lead_id="MINIMAL",
            company_name="Minimal Corp",
            industry="Unknown",
            job_title="Employee",
            company_size=None,
            lead_source="website",
            created_date=datetime.now(),
            email_opens=0,
            website_visits=0,
            form_submissions=0,
            meeting_scheduled=False,
            meeting_attended=False,
            response_time_hours=None
        )
        
        engineer = FeatureEngineer()
        features = engineer.engineer_features(minimal_lead)
        
        # Should still produce valid features
        assert len(features) > 0
        assert all(0 <= value <= 1 for value in features.values())
        
        # Test with very old lead
        old_lead = LeadData(
            lead_id="OLD",
            company_name="Old Corp",
            industry="Technology",
            job_title="Manager",
            company_size=100,
            lead_source="website",
            created_date=datetime.now() - timedelta(days=365),  # 1 year old
            email_opens=1,
            website_visits=1,
            form_submissions=0
        )
        
        features_old = engineer.engineer_features(old_lead)
        
        # Lead age decay should be very low
        assert features_old['lead_age_decay'] < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 