# AI Lead Scoring Engine - Cleardeals Assignment

A comprehensive solution for Cleardeals' AI Lead Scoring Engine, designed to predict high-intent prospects and achieve 2-3x conversion lift through real-time intent scoring.

## 🎯 Project Overview

**Problem**: Brokers waste significant time on low-intent leads, reducing sales efficiency and conversion rates.

**Solution**: An AI-powered Lead Scoring Engine that provides real-time "Intent Score" predictions via CRM/WhatsApp integration with sub-300ms latency.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Feature Store  │───▶│  ML Pipeline    │
│                 │    │                 │    │                 │
│ • CRM Data      │    │ • Real-time     │    │ • Gradient      │
│ • Website       │    │ • Batch         │    │   Boosting      │
│ • Social Media  │    │ • Caching       │    │ • LLM Refranker │
│ • Email         │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   FastAPI       │    │   Monitoring    │
                       │   + Redis       │    │   & Drift       │
                       │                 │    │   Detection     │
                       │ • <300ms        │    │                 │
                       │ • Scalable      │    │ • Data Drift    │
                       │ • Cached        │    │ • Model Drift   │
                       └─────────────────┘    └─────────────────┘
```

## 📋 Assignment Solutions

### 1. Data & Feature Brainstorming (1.5h)

#### Data Sources
- **CRM Data**: Lead source, company size, industry, contact frequency
- **Website Behavior**: Page views, time on site, form submissions, download patterns
- **Email Engagement**: Open rates, click-through rates, response times
- **Social Media**: LinkedIn profile completeness, company following, engagement
- **Demographic**: Location, job title, company revenue, decision-making authority

#### Predictive Features
**Behavioral Features:**
- `engagement_score`: Weighted combination of email opens, clicks, and website visits
- `response_velocity`: Time between initial contact and first response
- `interaction_frequency`: Number of touchpoints in last 30 days

**Demographic Features:**
- `decision_maker_probability`: Based on job title and company hierarchy
- `company_maturity_score`: Company age, employee count, funding stage
- `industry_affinity`: Historical conversion rates by industry

**Temporal Features:**
- `seasonal_intent`: Day of week, month, quarter patterns
- `time_to_quarter_end`: Proximity to sales quarter deadlines
- `lead_age_score`: Time since lead generation with decay function

### 2. Data Quality & Compliance (1.5h)

#### Data Quality Challenges

**Challenge 1: Incomplete Lead Information**
- **Problem**: 40-60% of leads lack complete demographic data
- **Impact**: Reduced model accuracy and feature availability
- **Mitigation**: 
  - Implement progressive profiling with smart defaults
  - Use external data enrichment APIs (Clearbit, ZoomInfo)
  - Create synthetic features from available data patterns

**Challenge 2: Noisy Behavioral Data**
- **Problem**: Website tracking inconsistencies and bot traffic
- **Impact**: False positive engagement signals
- **Mitigation**:
  - Implement bot detection using user agent analysis
  - Use session-based aggregation to reduce noise
  - Apply statistical outlier detection for engagement metrics

#### Compliance Issues

**Issue 1: DPDP Compliance for PII**
- **Challenge**: Storing and processing personal data in India
- **Solution**:
  - Implement data localization with AWS Mumbai region
  - Use encryption at rest and in transit (AES-256)
  - Implement data retention policies with automatic deletion
  - Create consent management system with granular permissions

**Issue 2: Consent-First Data Collection**
- **Challenge**: Ensuring explicit consent for all data processing
- **Solution**:
  - Implement cookie consent banners with granular options
  - Create transparent data usage policies
  - Provide easy opt-out mechanisms
  - Regular consent audits and renewal processes

### 3. Model Choice & Metrics (1.5h)

#### Why Gradient Boosted Trees + LLM Refranker?

**Gradient Boosted Trees (XGBoost/LightGBM):**
- **Handles Mixed Data Types**: Naturally handles numerical, categorical, and sparse features
- **Feature Importance**: Provides interpretable feature rankings for business stakeholders
- **Robust to Outliers**: Less sensitive to data quality issues common in sales data
- **Fast Inference**: Sub-100ms prediction times suitable for real-time scoring
- **Handles Non-linear Relationships**: Captures complex interactions in lead behavior

**LLM Refranker:**
- **Contextual Understanding**: Leverages unstructured data (email content, notes)
- **Explainability**: Provides natural language explanations for predictions
- **Adaptability**: Can incorporate new information without retraining
- **Business Alignment**: Generates insights that resonate with sales teams

#### Technical Metrics
1. **AUC-ROC (0.85+ target)**: Measures model's ability to distinguish between high/low intent leads
2. **Precision at Top 20% (0.70+ target)**: Ensures high-quality leads in top scoring tier

#### Business Metrics
1. **Conversion Lift (2-3x target)**: Increase in conversion rate for high-scored leads
2. **Time-to-Conversion Reduction (40% target)**: Faster sales cycles for scored leads

### 4. Improving Lift & Adoption (1.5h)

#### Model Performance Enhancements

**Enhancement 1: Ensemble Learning with Domain Expertise**
- Combine multiple models (GBM, Neural Network, Rule-based)
- Incorporate domain expert rules as additional features
- Use weighted voting based on historical performance
- **Expected Impact**: 15-20% improvement in AUC-ROC

**Enhancement 2: Dynamic Feature Engineering**
- Implement automated feature selection using SHAP values
- Create interaction features based on business logic
- Use time-series features for temporal patterns
- **Expected Impact**: 10-15% improvement in precision

#### Sales Team Adoption Strategies

**Strategy 1: Gamification & Transparency**
- Create leaderboards for lead scoring accuracy
- Provide detailed explanations for each score
- Show historical performance of scored leads
- **Implementation**: Dashboard with real-time scoring insights

**Strategy 2: Integration & Workflow Optimization**
- Seamless CRM integration with one-click scoring
- Automated lead routing based on scores
- Personalized follow-up suggestions
- **Implementation**: Chrome extension + API integration

### 5. Real-Time Architecture & Scale (1h)

#### Architecture Components

**API Layer (FastAPI + Redis):**
```python
# FastAPI endpoint with Redis caching
@app.post("/score-lead")
async def score_lead(lead_data: LeadData):
    cache_key = f"lead_score:{lead_data.lead_id}"
    
    # Check cache first
    cached_score = await redis.get(cache_key)
    if cached_score:
        return {"score": float(cached_score), "source": "cache"}
    
    # Real-time scoring
    features = feature_engineer(lead_data)
    score = model.predict(features)
    
    # Cache for 5 minutes
    await redis.setex(cache_key, 300, str(score))
    
    return {"score": score, "source": "model"}
```

**Scalability Considerations:**
- **Horizontal Scaling**: Load balancer with multiple API instances
- **Database Optimization**: Read replicas, connection pooling
- **Caching Strategy**: Multi-level caching (Redis + CDN)
- **Rate Limiting**: Protect against abuse and ensure fair usage

#### Performance Optimizations
- **Model Optimization**: ONNX conversion for faster inference
- **Feature Pre-computation**: Batch processing for common features
- **Connection Pooling**: Efficient database and external API connections
- **Monitoring**: Real-time latency and throughput tracking

### 6. Monitoring & Continuous Learning (1h)

#### Drift Detection Mechanisms

**Mechanism 1: Statistical Drift Detection**
```python
class DataDriftMonitor:
    def __init__(self):
        self.reference_distribution = None
        self.drift_threshold = 0.05
    
    def detect_drift(self, current_data):
        # KS test for distribution changes
        ks_statistic, p_value = ks_2samp(
            self.reference_distribution, 
            current_data
        )
        
        if p_value < self.drift_threshold:
            self.trigger_retraining()
```

**Mechanism 2: Model Performance Monitoring**
- Track AUC-ROC degradation over time
- Monitor prediction distribution shifts
- Alert on significant performance drops
- Automatic model retraining triggers

#### Continuous Learning Pipeline
- **Daily Retraining**: Automated pipeline with new data
- **A/B Testing**: Gradual rollout of new models
- **Performance Tracking**: Real-time metrics dashboard
- **Rollback Capability**: Quick reversion to previous models

## 🚀 Implementation Roadmap

### Phase 1: MVP (4 weeks)
- Basic feature engineering pipeline
- Simple GBM model with core features
- FastAPI endpoint with Redis caching
- Basic monitoring dashboard

### Phase 2: Enhancement (6 weeks)
- LLM refranker integration
- Advanced feature engineering
- Comprehensive monitoring
- Sales team integration

### Phase 3: Scale (8 weeks)
- Production deployment
- Performance optimization
- Advanced drift detection
- Full CRM integration

## 📊 Expected Outcomes

- **2-3x conversion lift** for high-scored leads
- **<300ms response time** for real-time scoring
- **40% reduction** in time-to-conversion
- **95%+ uptime** for production system
- **80%+ adoption rate** among sales teams

## 🔧 Technical Stack

- **ML Framework**: XGBoost, Transformers (Hugging Face)
- **API Framework**: FastAPI, Redis
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Monitoring**: Prometheus, Grafana
- **Deployment**: Docker, Kubernetes
- **Cloud**: AWS (Mumbai region for compliance)

## 📝 License

MIT License - This project is open source and available for public use.

---

**Author**: Shubham Kumar  
**LinkedIn**: https://www.linkedin.com/in/hasteturtle 

**GitHub**: https://github.com/newturk  
**Submission Date**: July 08, 2025
