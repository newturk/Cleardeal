"""
ML Pipeline for AI Lead Scoring Engine
Handles model training, evaluation, and deployment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import joblib
import logging
from datetime import datetime
import json
import os

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

from .feature_engineering import FeatureEngineer, LeadData

logger = logging.getLogger(__name__)


class LeadScoringModel:
    """Main model class for lead scoring"""
    
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_metadata = {}
        
    def train(self, training_data: pd.DataFrame, target_column: str = "converted") -> Dict[str, float]:
        """
        Train the lead scoring model
        
        Args:
            training_data: DataFrame with lead features and target
            target_column: Name of the target column
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Starting model training with {len(training_data)} samples")
        
        # Prepare features
        X, y = self._prepare_training_data(training_data, target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize and train model
        self.model = self._create_model()
        
        if self.model_type == "xgboost":
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                early_stopping_rounds=50,
                verbose=False
            )
        elif self.model_type == "lightgbm":
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        else:
            self.model.fit(X_train, y_train)
        
        # Evaluate model
        metrics = self._evaluate_model(X_test, y_test)
        
        # Store metadata
        self.model_metadata = {
            "training_date": datetime.now().isoformat(),
            "model_type": self.model_type,
            "n_samples": len(training_data),
            "n_features": len(self.feature_names),
            "metrics": metrics
        }
        
        logger.info(f"Model training completed. AUC: {metrics['auc']:.4f}")
        return metrics
    
    def predict(self, lead_data: LeadData) -> Tuple[float, Dict[str, float]]:
        """
        Predict lead score and feature importance
        
        Args:
            lead_data: LeadData object with lead information
            
        Returns:
            Tuple of (score, feature_importance_dict)
        """
        # Engineer features
        features = self.feature_engineer.engineer_features(lead_data)
        
        # Convert to feature vector
        feature_vector = self._features_to_vector(features)
        
        # Make prediction
        if self.model_type == "xgboost":
            score = self.model.predict_proba(feature_vector.reshape(1, -1))[0][1]
        else:
            score = self.model.predict_proba(feature_vector.reshape(1, -1))[0][1]
        
        # Get feature importance
        feature_importance = self._get_feature_importance(features)
        
        return float(score), feature_importance
    
    def _prepare_training_data(self, data: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data by engineering features"""
        features_list = []
        targets = []
        
        for _, row in data.iterrows():
            # Convert DataFrame row to LeadData object
            lead_data = self._row_to_lead_data(row)
            
            # Engineer features
            features = self.feature_engineer.engineer_features(lead_data)
            features_list.append(features)
            targets.append(row[target_column])
        
        # Convert to feature matrix
        feature_matrix = self._features_to_matrix(features_list)
        
        return feature_matrix, np.array(targets)
    
    def _row_to_lead_data(self, row: pd.Series) -> LeadData:
        """Convert DataFrame row to LeadData object"""
        return LeadData(
            lead_id=str(row.get('lead_id', '')),
            company_name=str(row.get('company_name', '')),
            industry=str(row.get('industry', '')),
            job_title=str(row.get('job_title', '')),
            company_size=row.get('company_size'),
            lead_source=str(row.get('lead_source', '')),
            created_date=pd.to_datetime(row.get('created_date', datetime.now())),
            email_opens=int(row.get('email_opens', 0)),
            website_visits=int(row.get('website_visits', 0)),
            form_submissions=int(row.get('form_submissions', 0)),
            meeting_scheduled=bool(row.get('meeting_scheduled', False)),
            meeting_attended=bool(row.get('meeting_attended', False)),
            response_time_hours=row.get('response_time_hours'),
            location=row.get('location'),
            funding_stage=row.get('funding_stage')
        )
    
    def _features_to_matrix(self, features_list: List[Dict[str, float]]) -> np.ndarray:
        """Convert list of feature dictionaries to feature matrix"""
        if not features_list:
            return np.array([])
        
        # Get all unique feature names
        all_features = set()
        for features in features_list:
            all_features.update(features.keys())
        
        self.feature_names = sorted(list(all_features))
        
        # Create feature matrix
        feature_matrix = np.zeros((len(features_list), len(self.feature_names)))
        
        for i, features in enumerate(features_list):
            for j, feature_name in enumerate(self.feature_names):
                feature_matrix[i, j] = features.get(feature_name, 0.0)
        
        return feature_matrix
    
    def _features_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dictionary to feature vector"""
        feature_vector = np.zeros(len(self.feature_names))
        
        for i, feature_name in enumerate(self.feature_names):
            feature_vector[i] = features.get(feature_name, 0.0)
        
        return feature_vector
    
    def _create_model(self):
        """Create model based on model_type"""
        if self.model_type == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='auc'
            )
        elif self.model_type == "lightgbm":
            return lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
        else:
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
    
    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'auc': roc_auc_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        # Calculate precision at top 20%
        top_20_idx = np.argsort(y_pred_proba)[-int(len(y_pred_proba) * 0.2):]
        precision_top_20 = precision_score(y_test[top_20_idx], y_pred[top_20_idx])
        metrics['precision_top_20'] = precision_top_20
        
        return metrics
    
    def _get_feature_importance(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get feature importance for prediction explanation"""
        if self.model is None:
            return {}
        
        # Get model feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
        else:
            # Fallback to feature weights
            importance_scores = np.ones(len(self.feature_names)) / len(self.feature_names)
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, feature_name in enumerate(self.feature_names):
            if feature_name in features:
                feature_importance[feature_name] = float(importance_scores[i])
        
        return feature_importance
    
    def save_model(self, filepath: str):
        """Save model and metadata"""
        model_data = {
            'model': self.model,
            'feature_engineer': self.feature_engineer,
            'feature_names': self.feature_names,
            'model_metadata': self.model_metadata,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model and metadata"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_engineer = model_data['feature_engineer']
        self.feature_names = model_data['feature_names']
        self.model_metadata = model_data['model_metadata']
        self.model_type = model_data['model_type']
        
        logger.info(f"Model loaded from {filepath}")


class ModelEnsemble:
    """Ensemble of multiple models for improved performance"""
    
    def __init__(self, models: List[LeadScoringModel], weights: List[float] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(models):
            raise ValueError("Number of weights must match number of models")
    
    def predict(self, lead_data: LeadData) -> Tuple[float, Dict[str, float]]:
        """Get ensemble prediction"""
        predictions = []
        feature_importances = []
        
        for model in self.models:
            score, importance = model.predict(lead_data)
            predictions.append(score)
            feature_importances.append(importance)
        
        # Weighted average of predictions
        ensemble_score = sum(p * w for p, w in zip(predictions, self.weights))
        
        # Average feature importance
        ensemble_importance = {}
        for importance_dict in feature_importances:
            for feature, importance in importance_dict.items():
                if feature not in ensemble_importance:
                    ensemble_importance[feature] = []
                ensemble_importance[feature].append(importance)
        
        # Average importance scores
        for feature in ensemble_importance:
            ensemble_importance[feature] = np.mean(ensemble_importance[feature])
        
        return ensemble_score, ensemble_importance


class ModelEvaluator:
    """Comprehensive model evaluation and comparison"""
    
    @staticmethod
    def cross_validate_model(model: LeadScoringModel, data: pd.DataFrame, 
                           target_column: str = "converted", cv_folds: int = 5) -> Dict[str, List[float]]:
        """Perform cross-validation"""
        X, y = model._prepare_training_data(data, target_column)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model._create_model(), X, y, cv=cv_folds, scoring='roc_auc')
        
        return {
            'cv_scores': cv_scores.tolist(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    @staticmethod
    def compare_models(models: Dict[str, LeadScoringModel], 
                      test_data: pd.DataFrame, 
                      target_column: str = "converted") -> pd.DataFrame:
        """Compare multiple models"""
        results = []
        
        for model_name, model in models.items():
            X_test, y_test = model._prepare_training_data(test_data, target_column)
            metrics = model._evaluate_model(X_test, y_test)
            
            results.append({
                'model': model_name,
                **metrics
            })
        
        return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    # Create sample training data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'lead_id': [f'L{i:05d}' for i in range(n_samples)],
        'company_name': [f'Company{i}' for i in range(n_samples)],
        'industry': np.random.choice(['Technology', 'Finance', 'Healthcare'], n_samples),
        'job_title': np.random.choice(['CEO', 'CTO', 'Manager', 'Engineer'], n_samples),
        'company_size': np.random.randint(10, 1000, n_samples),
        'lead_source': np.random.choice(['website', 'referral', 'event'], n_samples),
        'created_date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'email_opens': np.random.poisson(3, n_samples),
        'website_visits': np.random.poisson(5, n_samples),
        'form_submissions': np.random.poisson(1, n_samples),
        'meeting_scheduled': np.random.choice([True, False], n_samples, p=[0.3, 0.7]),
        'meeting_attended': np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
        'response_time_hours': np.random.exponential(24, n_samples),
        'converted': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    })
    
    # Train model
    model = LeadScoringModel(model_type="xgboost")
    metrics = model.train(sample_data)
    
    print("Training Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Test prediction
    test_lead = LeadData(
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
        response_time_hours=2.0
    )
    
    score, importance = model.predict(test_lead)
    print(f"\nPredicted Score: {score:.4f}")
    print("Top Feature Importance:")
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feature, imp in sorted_importance[:5]:
        print(f"  {feature}: {imp:.4f}") 