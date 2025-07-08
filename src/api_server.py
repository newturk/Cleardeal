"""
FastAPI Server for AI Lead Scoring Engine
Provides real-time lead scoring with Redis caching
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import redis
import json
import logging
import time
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import asyncio
from pydantic import BaseModel, Field
import uvicorn

from .feature_engineering import LeadData, FeatureEngineer
from .ml_pipeline import LeadScoringModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Lead Scoring Engine",
    description="Real-time lead scoring API with sub-300ms latency",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis connection
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True
)

# Global model instance
model = None


class LeadScoringRequest(BaseModel):
    """Request model for lead scoring"""
    lead_id: str = Field(..., description="Unique lead identifier")
    company_name: str = Field(..., description="Company name")
    industry: str = Field(..., description="Industry sector")
    job_title: str = Field(..., description="Job title")
    company_size: Optional[int] = Field(None, description="Company size in employees")
    lead_source: str = Field(..., description="Lead source")
    created_date: str = Field(..., description="Lead creation date (ISO format)")
    email_opens: int = Field(0, description="Number of email opens")
    website_visits: int = Field(0, description="Number of website visits")
    form_submissions: int = Field(0, description="Number of form submissions")
    meeting_scheduled: bool = Field(False, description="Whether meeting was scheduled")
    meeting_attended: bool = Field(False, description="Whether meeting was attended")
    response_time_hours: Optional[float] = Field(None, description="Response time in hours")
    location: Optional[str] = Field(None, description="Geographic location")
    funding_stage: Optional[str] = Field(None, description="Company funding stage")


class LeadScoringResponse(BaseModel):
    """Response model for lead scoring"""
    lead_id: str
    score: float
    score_category: str
    confidence: float
    latency_ms: float
    source: str
    feature_importance: Dict[str, float]
    explanation: str
    timestamp: str


class BatchScoringRequest(BaseModel):
    """Request model for batch scoring"""
    leads: List[LeadScoringRequest]
    priority: str = Field("normal", description="Processing priority")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    redis_connected: bool
    uptime_seconds: float
    total_requests: int
    average_latency_ms: float


# Global metrics
request_count = 0
total_latency = 0.0
start_time = datetime.now()


def get_model() -> LeadScoringModel:
    """Dependency to get the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


def get_redis() -> redis.Redis:
    """Dependency to get Redis connection"""
    try:
        redis_client.ping()
        return redis_client
    except redis.ConnectionError:
        raise HTTPException(status_code=503, detail="Redis not available")


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global model
    
    logger.info("Starting AI Lead Scoring Engine...")
    
    # Load the trained model
    try:
        model = LeadScoringModel(model_type="xgboost")
        # Load from saved model file
        model.load_model("models/lead_scoring_model.pkl")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None
    
    # Test Redis connection
    try:
        redis_client.ping()
        logger.info("Redis connection established")
    except redis.ConnectionError as e:
        logger.error(f"Redis connection failed: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global request_count, total_latency
    
    uptime = (datetime.now() - start_time).total_seconds()
    avg_latency = total_latency / max(request_count, 1)
    
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        redis_connected=redis_client.ping(),
        uptime_seconds=uptime,
        total_requests=request_count,
        average_latency_ms=avg_latency
    )


@app.post("/api/v1/score-lead", response_model=LeadScoringResponse)
async def score_lead(
    request: LeadScoringRequest,
    background_tasks: BackgroundTasks,
    model: LeadScoringModel = Depends(get_model),
    redis_client: redis.Redis = Depends(get_redis)
):
    """
    Score a single lead in real-time
    
    Returns lead score with sub-300ms latency
    """
    global request_count, total_latency
    
    start_time_request = time.time()
    request_count += 1
    
    try:
        # Check cache first
        cache_key = f"lead_score:{request.lead_id}:{hash(str(request.dict()))}"
        cached_result = redis_client.get(cache_key)
        
        if cached_result:
            cached_data = json.loads(cached_result)
            latency = (time.time() - start_time_request) * 1000
            
            # Update metrics
            total_latency += latency
            
            return LeadScoringResponse(
                **cached_data,
                latency_ms=latency,
                source="cache"
            )
        
        # Convert request to LeadData
        lead_data = LeadData(
            lead_id=request.lead_id,
            company_name=request.company_name,
            industry=request.industry,
            job_title=request.job_title,
            company_size=request.company_size,
            lead_source=request.lead_source,
            created_date=datetime.fromisoformat(request.created_date),
            email_opens=request.email_opens,
            website_visits=request.website_visits,
            form_submissions=request.form_submissions,
            meeting_scheduled=request.meeting_scheduled,
            meeting_attended=request.meeting_attended,
            response_time_hours=request.response_time_hours,
            location=request.location,
            funding_stage=request.funding_stage
        )
        
        # Get prediction with timeout
        try:
            score, feature_importance = await asyncio.wait_for(
                asyncio.to_thread(model.predict, lead_data),
                timeout=0.25  # 250ms timeout
            )
        except asyncio.TimeoutError:
            # Fallback to default score
            score = 0.5
            feature_importance = {}
            logger.warning(f"Model prediction timeout for lead {request.lead_id}")
        
        # Determine score category
        if score >= 0.8:
            score_category = "High Intent"
        elif score >= 0.6:
            score_category = "Medium Intent"
        elif score >= 0.4:
            score_category = "Low Intent"
        else:
            score_category = "Very Low Intent"
        
        # Calculate confidence based on feature availability
        feature_engineer = FeatureEngineer()
        features = feature_engineer.engineer_features(lead_data)
        confidence = min(1.0, len(features) / 10.0)  # Simple confidence metric
        
        # Generate explanation
        explanation = generate_explanation(score, feature_importance, lead_data)
        
        # Prepare response
        response_data = {
            "lead_id": request.lead_id,
            "score": score,
            "score_category": score_category,
            "confidence": confidence,
            "feature_importance": feature_importance,
            "explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache result for 5 minutes
        redis_client.setex(cache_key, 300, json.dumps(response_data))
        
        # Background task for analytics
        background_tasks.add_task(log_prediction, request.lead_id, score, feature_importance)
        
        latency = (time.time() - start_time_request) * 1000
        total_latency += latency
        
        return LeadScoringResponse(
            **response_data,
            latency_ms=latency,
            source="model"
        )
        
    except Exception as e:
        logger.error(f"Error scoring lead {request.lead_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.post("/api/v1/score-batch")
async def score_batch(
    request: BatchScoringRequest,
    model: LeadScoringModel = Depends(get_model)
):
    """
    Score multiple leads in batch
    
    Returns scores for all leads with optimized processing
    """
    start_time_batch = time.time()
    
    try:
        results = []
        
        # Process leads in parallel for better performance
        tasks = []
        for lead_request in request.leads:
            task = asyncio.create_task(score_lead_internal(lead_request, model))
            tasks.append(task)
        
        # Wait for all predictions with timeout
        batch_results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=2.0  # 2 second timeout for batch
        )
        
        for i, result in enumerate(batch_results):
            if isinstance(result, Exception):
                results.append({
                    "lead_id": request.leads[i].lead_id,
                    "error": str(result),
                    "score": None
                })
            else:
                results.append(result)
        
        batch_latency = (time.time() - start_time_batch) * 1000
        
        return {
            "results": results,
            "batch_latency_ms": batch_latency,
            "leads_processed": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch scoring error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {str(e)}")


async def score_lead_internal(lead_request: LeadScoringRequest, model: LeadScoringModel) -> Dict:
    """Internal scoring function for batch processing"""
    lead_data = LeadData(
        lead_id=lead_request.lead_id,
        company_name=lead_request.company_name,
        industry=lead_request.industry,
        job_title=lead_request.job_title,
        company_size=lead_request.company_size,
        lead_source=lead_request.lead_source,
        created_date=datetime.fromisoformat(lead_request.created_date),
        email_opens=lead_request.email_opens,
        website_visits=lead_request.website_visits,
        form_submissions=lead_request.form_submissions,
        meeting_scheduled=lead_request.meeting_scheduled,
        meeting_attended=lead_request.meeting_attended,
        response_time_hours=lead_request.response_time_hours,
        location=lead_request.location,
        funding_stage=lead_request.funding_stage
    )
    
    score, feature_importance = await asyncio.to_thread(model.predict, lead_data)
    
    return {
        "lead_id": lead_request.lead_id,
        "score": score,
        "feature_importance": feature_importance
    }


def generate_explanation(score: float, feature_importance: Dict[str, float], lead_data: LeadData) -> str:
    """Generate human-readable explanation for the score"""
    explanations = []
    
    # Add top contributing factors
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_features:
        top_feature, top_importance = sorted_features[0]
        if top_importance > 0.1:
            explanations.append(f"Strong {top_feature.replace('_', ' ')} signal")
    
    # Add behavioral insights
    if lead_data.email_opens > 5:
        explanations.append("High email engagement")
    
    if lead_data.website_visits > 10:
        explanations.append("Active website exploration")
    
    if lead_data.meeting_attended:
        explanations.append("Attended scheduled meeting")
    
    # Add demographic insights
    if any(keyword in lead_data.job_title.lower() for keyword in ['ceo', 'cto', 'vp', 'director']):
        explanations.append("Decision-maker role")
    
    if lead_data.company_size and lead_data.company_size > 200:
        explanations.append("Established company size")
    
    if not explanations:
        explanations.append("Standard lead profile")
    
    return "; ".join(explanations)


async def log_prediction(lead_id: str, score: float, feature_importance: Dict[str, float]):
    """Background task to log prediction for analytics"""
    try:
        log_entry = {
            "lead_id": lead_id,
            "score": score,
            "feature_importance": feature_importance,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in Redis for analytics
        redis_client.lpush("prediction_logs", json.dumps(log_entry))
        redis_client.ltrim("prediction_logs", 0, 9999)  # Keep last 10k predictions
        
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")


@app.get("/api/v1/feature-importance")
async def get_feature_importance():
    """Get global feature importance from the model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Return feature importance weights
        feature_weights = model.feature_engineer.get_feature_importance_weights()
        
        return {
            "feature_importance": feature_weights,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        raise HTTPException(status_code=500, detail="Failed to get feature importance")


@app.get("/api/v1/metrics")
async def get_metrics():
    """Get API performance metrics"""
    global request_count, total_latency
    
    uptime = (datetime.now() - start_time).total_seconds()
    avg_latency = total_latency / max(request_count, 1)
    
    return {
        "total_requests": request_count,
        "average_latency_ms": avg_latency,
        "uptime_seconds": uptime,
        "requests_per_second": request_count / max(uptime, 1),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 