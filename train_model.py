#!/usr/bin/env python3
"""
Script to train a sample model for the AI Lead Scoring Engine
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
from src.ml_pipeline import LeadScoringModel
from src.feature_engineering import LeadData

def create_sample_data(n_samples=1000):
    """Create sample training data"""
    np.random.seed(42)
    
    # Generate sample leads
    leads = []
    for i in range(n_samples):
        # Random company size
        company_size = np.random.choice([10, 50, 100, 200, 500, 1000], p=[0.2, 0.3, 0.2, 0.15, 0.1, 0.05])
        
        # Random industry
        industry = np.random.choice(['Technology', 'Finance', 'Healthcare', 'Manufacturing'], p=[0.4, 0.3, 0.2, 0.1])
        
        # Random job title
        job_title = np.random.choice(['CEO', 'CTO', 'VP Engineering', 'Manager', 'Engineer'], p=[0.1, 0.15, 0.2, 0.3, 0.25])
        
        # Random lead source
        lead_source = np.random.choice(['website', 'referral', 'event', 'social'], p=[0.5, 0.2, 0.2, 0.1])
        
        # Random creation date (within last 30 days)
        created_date = datetime.now() - timedelta(days=np.random.randint(1, 30))
        
        # Random engagement metrics
        email_opens = np.random.poisson(3)
        website_visits = np.random.poisson(5)
        form_submissions = np.random.poisson(1)
        
        # Random meeting data
        meeting_scheduled = np.random.choice([True, False], p=[0.3, 0.7])
        meeting_attended = meeting_scheduled and np.random.choice([True, False], p=[0.8, 0.2])
        
        # Random response time
        response_time_hours = np.random.exponential(24) if np.random.random() > 0.3 else None
        
        # Random funding stage
        funding_stage = np.random.choice(['seed', 'series_a', 'series_b', 'series_c', 'public'], p=[0.3, 0.3, 0.2, 0.15, 0.05])
        
        # Create lead data
        lead = LeadData(
            lead_id=f"L{i:05d}",
            company_name=f"Company{i}",
            industry=industry,
            job_title=job_title,
            company_size=company_size,
            lead_source=lead_source,
            created_date=created_date,
            email_opens=email_opens,
            website_visits=website_visits,
            form_submissions=form_submissions,
            meeting_scheduled=meeting_scheduled,
            meeting_attended=meeting_attended,
            response_time_hours=response_time_hours,
            funding_stage=funding_stage
        )
        
        leads.append(lead)
    
    return leads

def create_training_dataframe(leads):
    """Convert leads to DataFrame for training"""
    data = []
    for lead in leads:
        # Create feature engineer to get features
        from src.feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()
        features = engineer.engineer_features(lead)
        
        # Add lead info
        row = {
            'lead_id': lead.lead_id,
            'company_name': lead.company_name,
            'industry': lead.industry,
            'job_title': lead.job_title,
            'company_size': lead.company_size,
            'lead_source': lead.lead_source,
            'created_date': lead.created_date,
            'email_opens': lead.email_opens,
            'website_visits': lead.website_visits,
            'form_submissions': lead.form_submissions,
            'meeting_scheduled': lead.meeting_scheduled,
            'meeting_attended': lead.meeting_attended,
            'response_time_hours': lead.response_time_hours,
            'funding_stage': lead.funding_stage
        }
        
        # Add engineered features
        row.update(features)
        
        # Create target variable (simplified conversion logic)
        conversion_prob = (
            features.get('engagement_velocity', 0) * 0.3 +
            features.get('decision_maker_probability', 0) * 0.3 +
            features.get('company_maturity_score', 0) * 0.2 +
            features.get('meeting_attendance_rate', 0) * 0.2
        )
        
        # Add some noise and convert to binary
        conversion_prob += np.random.normal(0, 0.1)
        converted = 1 if conversion_prob > 0.5 else 0
        
        row['converted'] = converted
        data.append(row)
    
    return pd.DataFrame(data)

def main():
    """Main training function"""
    print("Creating sample training data...")
    leads = create_sample_data(1000)
    
    print("Converting to DataFrame...")
    training_data = create_training_dataframe(leads)
    
    print(f"Training data shape: {training_data.shape}")
    print(f"Conversion rate: {training_data['converted'].mean():.3f}")
    
    # Create and train model
    print("Training model...")
    model = LeadScoringModel(model_type="xgboost")
    
    # Train the model
    metrics = model.train(training_data, target_column="converted")
    
    print("Training completed!")
    print("Model metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save the model
    model_path = "models/lead_scoring_model.pkl"
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Test the model
    print("\nTesting model with sample lead...")
    test_lead = LeadData(
        lead_id="TEST001",
        company_name="TechCorp Inc",
        industry="Technology",
        job_title="VP Engineering",
        company_size=200,
        lead_source="website",
        created_date=datetime.now() - timedelta(days=5),
        email_opens=5,
        website_visits=10,
        form_submissions=2,
        meeting_scheduled=True,
        meeting_attended=True,
        response_time_hours=2.0,
        funding_stage="series_b"
    )
    
    score, importance = model.predict(test_lead)
    print(f"Test lead score: {score:.4f}")
    print("Top feature importance:")
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feature, imp in sorted_importance[:5]:
        print(f"  {feature}: {imp:.4f}")

if __name__ == "__main__":
    main() 