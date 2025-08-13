# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import os

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and data
classifier = None
sentence_model = None
faiss_index = None
investor_data = None

def load_models():
    """Load all AI models and data at startup"""
    global classifier, sentence_model, faiss_index, investor_data
    
    print(" Loading AI models...")
    
    try:
        # Load sentiment analysis model (existing)
        classifier = pipeline(
            "text-classification",
            model="./startup_sentiment_bert",  # your fine-tuned model path
            tokenizer="./startup_sentiment_bert"
        )
        print(" Sentiment analysis model loaded")
    except Exception as e:
        print(f" Warning: Could not load sentiment model: {e}")
        # Create a dummy classifier for testing
        classifier = None
    
    try:
        # Load Sentence-BERT for investor recommendations
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(" Sentence-BERT model loaded")
    except Exception as e:
        print(f" Error loading Sentence-BERT: {e}")
        return False
    
    try:
        # Load FAISS index
        faiss_index = faiss.read_index('investor_index.faiss')
        print(f" FAISS index loaded with {faiss_index.ntotal} investors")
    except Exception as e:
        print(f" Error loading FAISS index: {e}")
        print("Please run investor_embeddings.py first to create the index")
        return False
    
    try:
        # Load investor data
        with open('investor_data.pkl', 'rb') as f:
            investor_data = pickle.load(f)
        print(f" Investor data loaded: {len(investor_data['dataframe'])} investors")
    except Exception as e:
        print(f" Error loading investor data: {e}")
        return False
    
    return True

# Load models at startup
if not load_models():
    print(" Failed to load required models. Please check your setup.")

# Request/Response models
class PitchRequest(BaseModel):
    text: str

class InvestorRecommendation(BaseModel):
    investor_name: str
    domains: str
    preferred_stage: str
    ticket_size: str
    location: str
    description: str
    linkedin: str  # Added LinkedIn URL field
    match_score: float

class RecommendationResponse(BaseModel):
    recommendations: List[InvestorRecommendation]
    query_processed: str

class SentimentResponse(BaseModel):
    label: str
    score: float

# Existing sentiment analysis endpoint
@app.post("/predict-sentiment", response_model=SentimentResponse)
def predict_sentiment(request: PitchRequest):
    """
    Analyzes the sentiment of a startup pitch using fine-tuned BERT.
    Returns sentiment label and confidence score.
    """
    if not classifier:
        # Fallback for testing if model not available
        return SentimentResponse(
            label="Positive",
            score=0.85
        )
    
    try:
        result = classifier(request.text)[0]
        return SentimentResponse(
            label=result['label'],
            score=float(result['score'])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

# New investor recommendation endpoint
@app.post("/recommend-investors", response_model=RecommendationResponse)
def recommend_investors(request: PitchRequest):
    """
    Recommends top 5 investors based on semantic similarity between
    the pitch text and investor profiles using Sentence-BERT + FAISS.
    """
    
    # Check if all required components are loaded
    if not sentence_model or not faiss_index or not investor_data:
        raise HTTPException(
            status_code=503, 
            detail="Recommendation system not available. Please ensure models are loaded."
        )
    
    try:
        # Generate embedding for the pitch
        pitch_embedding = sentence_model.encode([request.text])
        
        # Normalize for cosine similarity (same as training data)
        faiss.normalize_L2(pitch_embedding.astype(np.float32))
        
        # Search for top 5 similar investors
        k = min(5, faiss_index.ntotal)  # In case we have fewer than 5 investors
        similarities, indices = faiss_index.search(pitch_embedding.astype(np.float32), k)
        
        # Prepare recommendations
        recommendations = []
        df = investor_data['dataframe']
        
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(df):  # Safety check
                investor = df.iloc[idx]
                
                recommendation = InvestorRecommendation(
                    investor_name=str(investor.get('investor_name', 'Unknown')),
                    domains=str(investor.get('preferred_domains', 'Not specified')),
                    preferred_stage=str(investor.get('funding_stages', 'Not specified')),
                    ticket_size=str(investor.get('investment_range_usd', 'Not specified')),
                    location=str(investor.get('location', 'Not specified')),
                    description=str(investor.get('description', 'No description available')),
                    linkedin=str(investor.get('linkedin', '')),  # Added LinkedIn URL
                    match_score=float(similarity)  # Cosine similarity score
                )
                recommendations.append(recommendation)
        
        return RecommendationResponse(
            recommendations=recommendations,
            query_processed=request.text[:100] + "..." if len(request.text) > 100 else request.text
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.get("/")
def read_root():
    """Health check and API information"""
    return {
        "message": "Startup Ecosystem Tracker API",
        "features": [
            "Pitch Sentiment Analysis (BERT)",
            "Investor Recommendations (Sentence-BERT + FAISS)"
        ],
        "endpoints": [
            "POST /predict-sentiment",
            "POST /recommend-investors"
        ],
        "status": {
            "sentiment_model": classifier is not None,
            "recommendation_system": all([sentence_model, faiss_index, investor_data])
        }
    }

@app.get("/health")
def health_check():
    """Detailed health check for all components"""
    return {
        "sentiment_analysis": classifier is not None,
        "sentence_bert": sentence_model is not None,
        "faiss_index": faiss_index is not None and faiss_index.ntotal > 0,
        "investor_data": investor_data is not None and len(investor_data.get('dataframe', [])) > 0,
        "total_investors": faiss_index.ntotal if faiss_index else 0
    }

from fastapi.responses import FileResponse

@app.get("/app")
def frontend():
    return FileResponse("index.html")
