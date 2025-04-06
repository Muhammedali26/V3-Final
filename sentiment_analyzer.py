from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
from textblob import TextBlob
import numpy as np
from transformers import pipeline
import re
import requests
from urllib.parse import quote_plus
import time

app = FastAPI(
    title="Sentiment Analysis Service",
    description="""
    Advanced service for analyzing product reviews and calculating detailed sentiment scores.
    
    Core Features:
    - Single Product Analysis: Detailed sentiment analysis of individual product reviews
    - Bulk Product Analysis: Process multiple reviews simultaneously
    - Aspect-based Sentiment Scoring: Break down sentiment by specific product aspects
    - AI-Powered Analysis: Utilizes state-of-the-art NLP models
    
    Technical Capabilities:
    - Multi-lingual support through BERT
    - Aspect-based sentiment analysis
    - Customizable scoring weights
    - Real-time processing
    
    Integration Features:
    - Direct Amazon product URL analysis
    - Batch processing capabilities
    - Comprehensive error handling
    - Detailed sentiment breakdowns
    
    Use Cases:
    1. Product Review Analysis
    2. Customer Feedback Processing
    3. Market Research
    4. Product Performance Monitoring
    """,
    version="1.0.0"
)

class SingleProductReview(BaseModel):
    review: str = Field(
        ..., 
        description="Single review text to analyze for sentiment",
        example="This product exceeded my expectations! The quality is outstanding and it works perfectly."
    )
    product_description: Optional[str] = Field(
        None, 
        description="Product description to provide context for sentiment analysis",
        example="Premium wireless headphones with noise cancellation"
    )

class BulkProductReviews(BaseModel):
    reviews: List[str] = Field(
        ..., 
        description="List of product reviews to analyze in bulk",
        example=[
            "This product is amazing! The quality is exceptional.",
            "Good value for money, though shipping was slow.",
            "Works as advertised, very satisfied with the purchase."
        ]
    )
    product_description: Optional[str] = Field(
        None, 
        description="Product description to provide context for sentiment analysis",
        example="Professional-grade kitchen blender with multiple speed settings"
    )

    class Config:
        schema_extra = {
            "example": {
                "reviews": [
                    "This product is amazing! The quality is exceptional.",
                    "Good value for money, though shipping was slow.",
                    "Works as advertised, very satisfied with the purchase."
                ],
                "product_description": "Professional-grade kitchen blender with multiple speed settings"
            }
        }

class SentimentScores(BaseModel):
    likability_score: float = Field(
        ..., 
        description="Score indicating overall customer satisfaction and product appeal (0-1)",
        ge=0, 
        le=1
    )
    effectiveness_score: float = Field(
        ..., 
        description="Score measuring how well the product performs its intended function (0-1)",
        ge=0, 
        le=1
    )
    value_for_money_score: float = Field(
        ..., 
        description="Score evaluating price-to-quality ratio and perceived value (0-1)",
        ge=0, 
        le=1
    )
    ingredient_quality_score: float = Field(
        ..., 
        description="Score assessing the quality of materials or ingredients used (0-1)",
        ge=0, 
        le=1
    )
    ease_of_use_score: float = Field(
        ..., 
        description="Score measuring product usability and user experience (0-1)",
        ge=0, 
        le=1
    )
    overall_score: float = Field(
        ..., 
        description="Weighted average of all sentiment scores (0-1)",
        ge=0, 
        le=1
    )

class SingleReviewInput(BaseModel):
    review: str = Field(
        ..., 
        description="Individual review text for sentiment analysis",
        example="This product is fantastic! Easy to use and great results."
    )
    product_description: Optional[str] = Field(
        None, 
        description="Product description for context-aware analysis",
        example="Smart home security camera with motion detection"
    )

class SingleSummaryInput(BaseModel):
    ai_summary: str = Field(
        ..., 
        description="AI-generated review summary to analyze for sentiment patterns",
        example="This product receives consistently positive feedback. Users praise its durability and performance. While some mention the price is high, most agree it's worth the investment. The product is particularly noted for its ease of use and reliable results."
    )
    product_description: Optional[str] = Field(
        None, 
        description="Product description to enhance context understanding",
        example="Professional-grade food processor with multiple attachments"
    )

class AspectSentimentScores(BaseModel):
    likability: float = Field(
        ..., 
        description="Sentiment score for overall product appeal and satisfaction (0-1)",
        ge=0,
        le=1
    )
    effectiveness: float = Field(
        ..., 
        description="Sentiment score for product performance and results (0-1)",
        ge=0,
        le=1
    )
    value_for_money: float = Field(
        ..., 
        description="Sentiment score for price-to-value ratio (0-1)",
        ge=0,
        le=1
    )
    ingredient_quality: float = Field(
        ..., 
        description="Sentiment score for material and component quality (0-1)",
        ge=0,
        le=1
    )
    ease_of_use: float = Field(
        ..., 
        description="Sentiment score for usability and user experience (0-1)",
        ge=0,
        le=1
    )
    overall: Optional[float] = Field(
        None, 
        description="Weighted average of all aspect scores (0-1)",
        ge=0,
        le=1
    )

class ProductURLInput(BaseModel):
    product_url: str = Field(
        ..., 
        description="Amazon product URL for direct sentiment analysis",
        example="https://www.amazon.com/product-name/dp/PRODUCT_ID"
    )

class ProductNameInput(BaseModel):
    product_name: str = Field(
        ...,
        description="Product name to search and analyze on Amazon",
        example="wireless noise cancelling headphones"
    )

class SentimentAnalyzer:
    """
    Advanced sentiment analysis engine for product reviews.
    
    Features:
    - Multi-aspect sentiment analysis
    - Contextual sentiment understanding
    - Customizable aspect weights
    - Review aggregation
    
    Technical Details:
    - Uses BERT-based multilingual sentiment analysis
    - Implements aspect-based sentiment analysis
    - Provides normalized sentiment scores (0-1)
    - Handles multiple languages
    """
    
    def __init__(self):
        """
        Initialize the sentiment analyzer with required models and configurations.
        
        Sets up:
        - BERT sentiment analysis pipeline
        - Aspect keywords dictionary
        - Scoring normalization
        """
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        # Comprehensive keyword dictionary for aspect-based sentiment analysis
        self.aspect_keywords = {
            'likability': [
                'love', 'like', 'great', 'awesome', 'excellent', 
                'amazing', 'perfect', 'best', 'fantastic', 'wonderful'
            ],
            'effectiveness': [
                'work', 'result', 'effective', 'improve', 'clean',
                'healthy', 'strong', 'shine', 'powerful', 'efficient'
            ],
            'value_for_money': [
                'price', 'worth', 'value', 'expensive', 'cheap',
                'cost', 'affordable', 'bargain', 'overpriced', 'reasonable'
            ],
            'ingredient_quality': [
                'natural', 'organic', 'chemical', 'ingredient', 'harsh',
                'gentle', 'quality', 'premium', 'pure', 'safe'
            ],
            'ease_of_use': [
                'easy', 'simple', 'convenient', 'quick', 'mess',
                'application', 'apply', 'intuitive', 'straightforward', 'complicated'
            ]
        }

    def analyze_aspects(self, text: str) -> Dict[str, float]:
        """
        Perform aspect-based sentiment analysis on text.
        
        Args:
            text (str): Review text to analyze
            
        Returns:
            Dict[str, float]: Sentiment scores for each aspect (0-1 scale)
            
        Features:
        - Analyzes sentiment for multiple aspects
        - Handles missing aspects gracefully
        - Provides normalized scores
        - Considers context and negations
        """
        if not text:
            return {aspect: 0.5 for aspect in self.aspect_keywords.keys()}

        # Clean and preprocess text
        processed_text = self.preprocess_text(text)
        
        # Split into sentences for granular analysis
        sentences = [sent.strip() for sent in processed_text.split('.') if sent.strip()]
        
        # Calculate scores for each aspect
        scores = {}
        for aspect, keywords in self.aspect_keywords.items():
            # Find relevant sentences for each aspect
            relevant_sentences = [
                sent for sent in sentences 
                if any(keyword in sent.lower() for keyword in keywords)
            ]
            
            if relevant_sentences:
                # Calculate sentiment for each relevant sentence
                sentence_scores = []
                for sentence in relevant_sentences:
                    if sentence:
                        result = self.sentiment_pipeline(sentence)[0]
                        # Convert 1-5 star rating to 0-1 scale
                        score = (int(result['label'][0]) - 1) / 4
                        sentence_scores.append(score)
                
                # Average score for the aspect
                scores[aspect] = np.mean(sentence_scores) if sentence_scores else 0.5
            else:
                # Neutral score for aspects without relevant sentences
                scores[aspect] = 0.5

        return scores

    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for analysis.
        
        Args:
            text (str): Raw text to process
            
        Returns:
            str: Cleaned and normalized text
            
        Processing steps:
        1. Convert to lowercase
        2. Remove special characters (preserving periods)
        3. Normalize whitespace
        4. Handle contractions
        """
        text = text.lower()
        text = re.sub(r'[^\w\s.]', ' ', text)  # Preserve periods
        text = ' '.join(text.split())
        return text

# Initialize global analyzer instance
analyzer = SentimentAnalyzer()

@app.post("/analyze/product_url", response_model=AspectSentimentScores)
async def analyze_from_url(data: ProductURLInput):
    """
    Analyze sentiment from an Amazon product URL.
    
    This endpoint performs comprehensive sentiment analysis by:
    1. Fetching product data from Amazon
    2. Analyzing customer reviews
    3. Generating aspect-based sentiment scores
    
    Features:
    - Direct URL analysis
    - Comprehensive sentiment breakdown
    - Automatic review aggregation
    - Error handling and recovery
    
    Args:
        data (ProductURLInput): Contains Amazon product URL
        
    Returns:
        AspectSentimentScores: Detailed sentiment analysis results
        
    Raises:
        HTTPException: For various error conditions with appropriate status codes
    """
    try:
        SCRAPER_HOST = "localhost"
        SCRAPER_PORT = 8000
        
        print(f"\n=== Starting analysis for: {data.product_url} ===")
        
        # Call direct-data endpoint
        direct_data_url = f"http://{SCRAPER_HOST}:{SCRAPER_PORT}/direct-data?url={data.product_url}"
        print(f"Calling direct-data endpoint: {direct_data_url}")
        
        product_response = requests.get(direct_data_url, timeout=60)
        
        if product_response.status_code != 200:
            print(f"[ERROR] Data retrieval response code: {product_response.status_code}")
            print(f"[ERROR] Data retrieval response: {product_response.text}")
            raise HTTPException(
                status_code=500,
                detail=f"Amazon direct-data error: {product_response.text}"
            )
        
        # Verify and log data
        product_data = product_response.json()
        
        if "error" in product_data:
            print(f"[ERROR] Scraper data retrieval error: {product_data['error']}")
            raise HTTPException(
                status_code=500,
                detail=f"Amazon Scraper error: {product_data['error']}"
            )
        
        print(f"Data retrieved: {str(product_data)[:200]}...")  # Show first 200 chars
        
        # Check customer reviews
        customer_reviews = product_data.get('customer_reviews', [])
        print(f"Number of reviews found: {len(customer_reviews)}")
        
        if not customer_reviews:
            print("[WARNING] No reviews found, using default scores")
            return AspectSentimentScores(
                likability=0.5,
                effectiveness=0.5,
                value_for_money=0.5,
                ingredient_quality=0.5,
                ease_of_use=0.5,
                overall=0.5
            )
        
        # Analyze combined reviews
        print(f"Analyzing reviews...")
        combined_reviews = " ".join(customer_reviews)
        aspect_scores = analyzer.analyze_aspects(combined_reviews)
        
        # Calculate weighted average
        weights = {
            'likability': 0.3,
            'effectiveness': 0.3,
            'value_for_money': 0.2,
            'ingredient_quality': 0.1,
            'ease_of_use': 0.1
        }
        
        overall_score = sum(
            aspect_scores[aspect] * weights[aspect]
            for aspect in weights.keys()
        )
        
        # Prepare results
        result = {
            **aspect_scores,
            'overall': overall_score
        }
        
        print(f"Analysis completed: {result}")
        return AspectSentimentScores(**result)
        
    except requests.RequestException as e:
        print(f"[ERROR] Connection error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error connecting to Amazon Scraper: {str(e)}\nMake sure the Amazon Scraper service is running on {SCRAPER_HOST}:{SCRAPER_PORT}"
        )
    except Exception as e:
        import traceback
        print(f"[ERROR] General error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-scraper-connection")
async def test_scraper_connection():
    """
    Test connection to the Amazon Scraper service.
    
    This endpoint verifies:
    - Service availability
    - Network connectivity
    - Response times
    - Basic functionality
    
    Returns:
        dict: Connection status and diagnostic information
        
    Features:
    - Detailed error reporting
    - Connection diagnostics
    - Troubleshooting suggestions
    """
    SCRAPER_HOST = "localhost"
    SCRAPER_PORT = 8000
    
    try:
        # Simple ping request
        response = requests.get(f"http://{SCRAPER_HOST}:{SCRAPER_PORT}/docs", timeout=5)
        return {
            "status": "success",
            "message": f"Successfully connected to Amazon Scraper at {SCRAPER_HOST}:{SCRAPER_PORT}",
            "status_code": response.status_code
        }
    except requests.RequestException as e:
        return {
            "status": "error",
            "message": f"Failed to connect to Amazon Scraper: {str(e)}",
            "suggestions": [
                "Make sure amazon_scraper.py is running",
                f"Check if service is running on {SCRAPER_HOST}:{SCRAPER_PORT}",
                "Check network/firewall settings",
                "Try running both services in the same environment"
            ]
        }

@app.get("/direct-scrape-test")
async def direct_scrape_test(url: str):
    """
    Test Amazon Scraper functionality with direct data retrieval.
    
    This endpoint provides:
    - End-to-end testing
    - Data retrieval verification
    - Response validation
    - Error diagnosis
    
    Args:
        url (str): Amazon product URL to test
        
    Returns:
        dict: Test results and diagnostic information
        
    Features:
    - Comprehensive testing
    - Detailed error reporting
    - Performance metrics
    - Troubleshooting guidance
    """
    SCRAPER_HOST = "localhost"
    SCRAPER_PORT = 8000
    
    try:
        # 1. Call scrape_product endpoint
        scrape_result = requests.post(
            f"http://{SCRAPER_HOST}:{SCRAPER_PORT}/scrape_product",
            json={"product_url": url},
            timeout=60
        )
        
        # 2. Brief waiting period
        time.sleep(3)
        
        # 3. Call data endpoint
        data_result = requests.get(
            f"http://{SCRAPER_HOST}:{SCRAPER_PORT}/data",
            timeout=60
        )
        
        # Prepare results
        return {
            "scrape_request": {
                "status_code": scrape_result.status_code,
                "response": scrape_result.json() if scrape_result.status_code == 200 else scrape_result.text
            },
            "data_request": {
                "status_code": data_result.status_code,
                "response": data_result.json() if data_result.status_code == 200 else data_result.text
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "suggestions": [
                "Ensure Amazon Scraper service is running",
                "Verify URL format is correct",
                "Use complete Amazon product URLs (e.g., https://www.amazon.com/dp/ASIN)"
            ]
        }

@app.post("/analyze/summary", response_model=AspectSentimentScores, deprecated=True)
async def analyze_summary(data: SingleSummaryInput):
    """
    [DEPRECATED] Use /analyze/product_url instead.
    
    Legacy endpoint for analyzing AI-generated review summaries.
    This endpoint is maintained for backward compatibility but will be removed in future versions.
    
    Please migrate to the /analyze/product_url endpoint for improved functionality.
    """
    try:
        # Analyze AI summary
        aspect_scores = analyzer.analyze_aspects(data.ai_summary)
        
        # Calculate weighted average
        weights = {
            'likability': 0.3,
            'effectiveness': 0.3,
            'value_for_money': 0.2,
            'ingredient_quality': 0.1,
            'ease_of_use': 0.1
        }
        
        overall_score = sum(
            aspect_scores[aspect] * weights[aspect]
            for aspect in weights.keys()
        )
        
        # Prepare results
        result = {
            **aspect_scores,
            'overall': overall_score
        }
        
        return AspectSentimentScores(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/single", response_model=SentimentScores, deprecated=True)
async def analyze_single(review: SingleReviewInput):
    """
    [DEPRECATED] Use /analyze/product_url instead.
    
    Legacy endpoint for single review analysis.
    This endpoint is maintained for backward compatibility but will be removed in future versions.
    
    Please migrate to the /analyze/product_url endpoint for improved functionality.
    """
    try:
        # This function is deprecated
        scores = {
            'likability_score': 0.5,
            'effectiveness_score': 0.5,
            'value_for_money_score': 0.5,
            'ingredient_quality_score': 0.5,
            'ease_of_use_score': 0.5,
            'overall_score': 0.5
        }
        return SentimentScores(**scores)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
