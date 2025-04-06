from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, UploadFile, File, Body
from pydantic import BaseModel, Field, validator
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Any, Optional, Tuple, Union
from difflib import SequenceMatcher
from datetime import datetime, timedelta
from hashlib import md5
from condition_scores import ConditionScores  # Add import for ConditionScores
from sentiment_analyzer import SentimentAnalyzer
from amazon_scraper import AmazonScraper  # Import the scraper class
import requests
import urllib.parse
import json
import os
import asyncio
from enum import Enum
import scipy.sparse
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import logging
import re

app = FastAPI(
    title="Recommendation Engine",
    description="Hair care product recommendation API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

class UserFeatures(BaseModel):
    features: List[float]
    top_n: int = 5
    wash_frequency: str

class ProductMapping:
    def __init__(self):
        self.name_to_id: Dict[str, int] = {}
        self.id_to_name: Dict[int, str] = {}
        self.next_id: int = 1

    def get_product_id(self, product_name: str) -> int:
        """Get or create product ID from name"""
        if product_name not in self.name_to_id:
            self.name_to_id[product_name] = self.next_id
            self.id_to_name[self.next_id] = product_name
            self.next_id += 1
        return self.name_to_id[product_name]

    def get_product_name(self, product_id: int) -> Optional[str]:
        """Get product name from ID"""
        return self.id_to_name.get(product_id)

class UserMapping:
    def __init__(self):
        self.user_info: Dict[str, Dict] = {}
        
    def generate_user_id(self, base_name: str) -> str:
        """Generate consistent user ID from base name"""
        # Example: "Nadine" -> "Nadine #1"
        existing_ids = [
            int(uid.split('#')[1]) 
            for uid in self.user_info.keys() 
            if uid.startswith(f"{base_name} #")
        ]
        next_id = max(existing_ids, default=0) + 1
        return f"{base_name} #{next_id}"
    
    def register_user(self, base_name: str, hair_type: Optional[str] = None) -> Tuple[str, bool]:
        """Register user and return (unique_id, is_new)"""
        unique_id = self.generate_user_id(base_name)
        is_new = unique_id not in self.user_info
        
        if is_new:
            self.user_info[unique_id] = {
                'base_name': base_name,
                'hair_type': hair_type,
                'registration_date': datetime.now()
            }
            
        return unique_id, is_new

class UserReview(BaseModel):
    product_name: str
    base_name: str
    hair_type: Optional[str]
    rating: float
    review_text: str
    purchase_date: Optional[datetime]

class UserPreferences(BaseModel):
    Name: str  # User's name
    Goal: Optional[str] = ""
    Issue: Optional[str] = ""
    Hair_Type: Optional[str] = ""
    Hair_Texture: Optional[str] = ""
    Hair_Behaviour: Optional[str] = ""
    Scalp_Feeling: Optional[str] = ""
    Wash_Hair: Optional[str] = ""
    Treatments: Optional[str] = "No not me"  # Default value for condition score calculation
    Scalp_Flaky: Optional[str] = "No Not Me"  # Default value for condition score calculation
    Oily_Scalp: Optional[str] = ""  # Required for sebum calculation
    Dry_Scalp: Optional[str] = ""  # Required for dry scalp calculation

    class Config:
        schema_extra = {
            "example": {
                "Name": "Clover L",
                "Goal": "Moisture retention",
                "Issue": "Not me",
                "Hair_Type": "Type 4b",
                "Hair_Texture": "Thin or Fine",
                "Hair_Behaviour": "B - Its bouncy and elastic",
                "Scalp_Feeling": "Sensitive",
                "Wash_Hair": "About once a month",
                "Treatments": "No not me",
                "Scalp_Flaky": "No Not Me",
                "Oily_Scalp": "3 - 4 Days",
                "Dry_Scalp": "Within hours"
            }
        }

class ModelUpdateConfig(BaseModel):
    retrain_knn: bool = True
    min_reviews: int = 100

class RecommendationCategories:
    SHAMPOO_MONTHLY = "Monthly Shampoo"
    SHAMPOO_WEEKLY = "Weekly Shampoo"
    CONDITIONER = "Conditioner"
    CONDITIONER_2 = "Conditioner 2"
    OTHER = "Other"

class ProductRecommendation(BaseModel):
    category: str
    product_name: str
    confidence_score: float
    sentiment_scores: Dict[str, float]
    usage_frequency: Optional[str]

class UserRecommendations(BaseModel):
    base_name: str
    unique_id: str
    hair_type: str
    recommendations: Dict[str, List[ProductRecommendation]]

class ProductSentimentCache:
    def __init__(self, cache_file="product_sentiments.json", refresh_interval_days=10):
        self.cache_file = cache_file
        self.refresh_interval = timedelta(days=refresh_interval_days)
        self.sentiments = {}
        self.last_updated = {}
        self.product_urls = {}
        
        # Load the cache
        self.load_cache()
    
    def load_cache(self):
        """Load sentiment cache from file"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.sentiments = data.get('sentiments', {})
                    self.last_updated = {k: datetime.fromisoformat(v) 
                                        for k, v in data.get('last_updated', {}).items()}
                    self.product_urls = data.get('product_urls', {})
                print(f"Loaded sentiment cache with {len(self.sentiments)} products")
            except Exception as e:
                print(f"Error loading sentiment cache: {str(e)}")
                self.sentiments = {}
                self.last_updated = {}
                self.product_urls = {}
    
    def save_cache(self, force=False):
        """
        Save cache to file
        
        Args:
            force: If True, immediately save and flush to disk
        """
        try:
            data = {
                'sentiments': self.sentiments,
                'last_updated': {k: v.isoformat() for k, v in self.last_updated.items()},
                'product_urls': self.product_urls
            }
            
            # Open and write the JSON file directly for immediate saving
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            # If force is True, ensure the file is saved
            if force:
                f.flush()
                os.fsync(f.fileno())
                
            print(f"Saved sentiment cache with {len(self.sentiments)} products")
        except Exception as e:
            print(f"Error saving sentiment cache: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def get_sentiment(self, product_name):
        """Get sentiment scores for a product"""
        if product_name in self.sentiments:
            return self.sentiments[product_name]
        return None
    
    def set_sentiment(self, product_name, sentiment_scores, product_url=None):
        """
        Set sentiment scores for a product
        """
        # Update sentiment data first
        self.sentiments[product_name] = sentiment_scores
        self.last_updated[product_name] = datetime.now()
        
        # If product_url is provided, update the URL as well
        if product_url:
            self.product_urls[product_name] = product_url
            
        # Save to file
        self.save_cache(force=True)
        print(f"Updated and saved sentiment scores for {product_name}")
    
    def needs_refresh(self, product_name):
        """Check if a product's sentiment data needs refreshing"""
        if product_name not in self.last_updated:
            return True
        
        now = datetime.now()
        return (now - self.last_updated[product_name]) > self.refresh_interval
    
    def get_url(self, product_name):
        """Get URL for a product"""
        return self.product_urls.get(product_name)
    
    def set_url(self, product_name, url):
        """
        Set URL for a product
        """
        self.product_urls[product_name] = url
        # Clear existing sentiment data when URL changes
        if product_name in self.sentiments:
            print(f"Clearing existing sentiment data for {product_name} due to URL change")
            self.sentiments.pop(product_name, None)
            
        if product_name in self.last_updated:
            self.last_updated.pop(product_name, None)
        
        # Save changes to file immediately
        self.save_cache(force=True)
        print(f"Updated URL for {product_name}: {url} and cleared existing sentiment data")

    def clear_sentiment(self, product_name):
        """
        Clear sentiment data for a specific product
        """
        print(f"Clearing sentiment data for {product_name}")
        if product_name in self.sentiments:
            self.sentiments.pop(product_name, None)
            
        if product_name in self.last_updated:
            self.last_updated.pop(product_name, None)
            
        # Save changes to file immediately
        self.save_cache(force=True)
        return self.get_url(product_name)
    
    def clear_all_sentiments(self):
        """
        Clear all sentiment data (preserves URLs)
        """
        product_count = len(self.sentiments)
        self.sentiments = {}
        self.last_updated = {}
        self.save_cache(force=True)
        print(f"Cleared all sentiment data for {product_count} products")

class RecommendationEngine:
    def __init__(self):
        # Base path for saving files
        self.base_path = "data"
        print(f"Base path: {self.base_path}")
        
        # Load score weights from file or use defaults
        self.score_weights = self.load_score_weights()
        
        # Initialize analyzers
        self.sentiment_analyzer = SentimentAnalyzer()
        self.condition_scorer = ConditionScores()
        
        print("Loading models and scalers...")
        self.load_models()
        
        # Feature information
        self.feature_columns = ['DRYNESS', 'DAMAGE', 'SENSITIVITY', 
                               'SEBUM_Oil', 'DRY_SCALP', 'FLAKES']
        
        # Try to load feature information 
        self.load_feature_info()
        
        # Load models and scalers
        try:
            print("Loading models and scalers...")
            self.scaler = self.load_pickle('scaler.pkl')
            self.vectorizer = self.load_pickle('vectorizer.pkl')
            
            # Try loading the existing nn_model if available
            try:
                self.knn_model = self.load_pickle('nn_model.pkl')
                print("NearestNeighbors model loaded successfully from nn_model.pkl")
            except Exception as e:
                print(f"Could not load nn_model.pkl: {str(e)}. Trying model.pkl...")
                try:
                    # Fall back to legacy model.pkl if nn_model.pkl is not available
                    self.knn_model = self.load_pickle('C:/Users/HP/Desktop/Project-master/Project-master/V3/model.pkl')
                    print("Legacy model loaded successfully from model.dpkl")
                except Exception as e2:
                    # If no model is available, create a new NearestNeighbors model
                    print(f"Could not load any model: {str(e2)}. Creating new NearestNeighbors model...")
                    from sklearn.neighbors import NearestNeighbors
                    self.knn_model = NearestNeighbors(
                        n_neighbors=10,
                        metric="cosine",
                        algorithm='brute',
                        n_jobs=-1  # Parallel processing
                    )
                    print("Created new NearestNeighbors model")
            
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
            # Create empty models as fallback
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import StandardScaler
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            print("Creating fallback models...")
            self.scaler = StandardScaler()
            self.vectorizer = TfidfVectorizer(
                stop_words="english", 
                ngram_range=(1, 2),
                max_features=2000,
                min_df=1,
                use_idf=True,
                sublinear_tf=True,
                analyzer='word',
                token_pattern=r'(?u)\b\w+\b'
            )
            self.knn_model = NearestNeighbors(
                n_neighbors=10,
                metric="cosine",
                algorithm='brute',
                n_jobs=-1
            )
            print("Fallback models created")
        
        # Add mappings
        self.product_mapping = ProductMapping()
        self.user_mapping = UserMapping()
        
        # Initialize DataFrames
        self.user_interactions = pd.DataFrame(columns=[
            'base_name', 'unique_id', 'product_name', 
            'hair_type', 'rating', 'review_text', 
            'purchase_date', 'interaction_date'
        ])
        
        self.user_preferences = pd.DataFrame(columns=[
            'base_name', 'unique_id', 'goal', 'issue', 
            'hair_type', 'hair_texture', 'wash_hair',
            'treatments', 'hair_behaviour', 'scalp_feeling',
            'scalp_flaky', 'oily_scalp', 'dry_scalp',
            'dryness', 'damage', 'sensitivity', 
            'sebum_oil', 'dry_scalp_score', 'flakes'
        ])

        # Update product category mapping
        self.product_categories = {
            'REC Shampoo MTH': RecommendationCategories.SHAMPOO_MONTHLY,
            'REC Shampoo WK': RecommendationCategories.SHAMPOO_WEEKLY,
            'REC Conditioner': RecommendationCategories.CONDITIONER,
            'REC Conditioner 2': RecommendationCategories.CONDITIONER_2,
            'REC Other ': RecommendationCategories.OTHER  # Note the trailing space!
        }

        # Add cache for product sentiment scores
        self.sentiment_cache = ProductSentimentCache("product_sentiments.json", refresh_interval_days=10)
        
    def load_pickle(self, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_amazon_data(self, product_name: str) -> Dict:
        """Get product data using AmazonScraper's direct-data endpoint"""
        try:
            # Scraper service information
            SCRAPER_HOST = "localhost"
            SCRAPER_PORT = 8000
            
            # URL encoding
            encoded_name = urllib.parse.quote_plus(product_name)
            
            # Use direct-data endpoint
            direct_data_url = f"http://{SCRAPER_HOST}:{SCRAPER_PORT}/direct-data?url=https://www.amazon.com/s?k={encoded_name}"
            
            print(f"Fetching data for product: {product_name}")
            response = requests.get(direct_data_url, timeout=60)
            
            if response.status_code != 200:
                print(f"Error fetching product data: Status code {response.status_code}")
                return None
            
            # Parse the data
            product_data = response.json()
            
            # Error checking
            if "error" in product_data:
                print(f"Error in Amazon Scraper response: {product_data['error']}")
                return None
            
            return product_data
            
        except Exception as e:
            print(f"Error fetching Amazon data: {str(e)}")
            return None

    def analyze_batch_sentiments(self, customer_reviews: List[str]) -> Dict[str, float]:
        """Perform sentiment analysis from customer reviews"""
        try:
            if not customer_reviews:
                return {aspect: 0.5 for aspect in self.score_weights['sentiment_weights'].keys()}
            
            # Combine and analyze reviews
            combined_reviews = " ".join(customer_reviews)
            
            # Use SentimentAnalyzer to analyze
            aspect_scores = self.sentiment_analyzer.analyze_aspects(combined_reviews)
            
            # Calculate overall score
            weights = self.score_weights['sentiment_weights']
            overall_score = sum(
                aspect_scores[aspect] * weights[aspect]
                for aspect in weights.keys()
            )
            
            aspect_scores['overall'] = overall_score
            return aspect_scores
            
        except Exception as e:
            print(f"Batch sentiment analysis error: {str(e)}")
            return {aspect: 0.5 for aspect in self.score_weights['sentiment_weights'].keys()}

    def calculate_final_score(self, knn_score: float, sentiment_data: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Calculate final score using weighted average of KNN and sentiment scores"""
        # Calculate average sentiment score
        sentiment_scores = {
            'likability': sentiment_data.get('likability', 0.65),
            'effectiveness': sentiment_data.get('effectiveness', 0.7),
            'value_for_money': sentiment_data.get('value_for_money', 0.6),
            'ingredient_quality': sentiment_data.get('ingredient_quality', 0.65),
            'ease_of_use': sentiment_data.get('ease_of_use', 0.7)
        }
        
        # Calculate overall sentiment score
        sentiment_score = sum(sentiment_scores.values()) / len(sentiment_scores)
        
        # Calculate final score using weights
        knn_weight = self.score_weights.get('knn_weight', 0.7)
        sentiment_weight = self.score_weights.get('sentiment_weight', 0.3)
        
        final_score = (knn_weight * knn_score) + (sentiment_weight * sentiment_score)
        
        return final_score, sentiment_scores

    def get_product_sentiment(self, product_name, force_refresh=False):
        """
        Get sentiment scores for a product (from cache or by calculating)
        
        Args:
            product_name: Product name
            force_refresh: If True, recalculate even if cached data exists
        """
        # Check if exists in cache
        cached_scores = self.sentiment_cache.get_sentiment(product_name)
        
        # Force refresh or refresh time check
        needs_refresh = force_refresh or self.sentiment_cache.needs_refresh(product_name)
        
        # If in cache and doesn't need refresh, return from cache
        if cached_scores and not needs_refresh:
            print(f"Using cached sentiment scores for {product_name}")
            return cached_scores
        
        # If force_refresh, log why we're refreshing
        if force_refresh and cached_scores:
            print(f"Forcing refresh of sentiment scores for {product_name}")
        
        # Check if we have a previously saved URL
        product_url = self.sentiment_cache.get_url(product_name)
        
        if product_url:
            try:
                # Fetch data directly from product URL
                print(f"Fetching sentiment scores for {product_name} from URL: {product_url}")
                
                SCRAPER_HOST = "localhost"
                SCRAPER_PORT = 8000
                
                # Use direct-data endpoint
                direct_data_url = f"http://{SCRAPER_HOST}:{SCRAPER_PORT}/direct-data?url={product_url}"
                print(f"Calling scraper endpoint: {direct_data_url}")
                
                response = requests.get(direct_data_url, timeout=60)
                
                if response.status_code == 200:
                    product_data = response.json()
                    
                    # Error checking
                    if "error" in product_data:
                        print(f"Error in scraper response: {product_data['error']}")
                    elif "customer_reviews" not in product_data:
                        print(f"No 'customer_reviews' field in response: {str(product_data)[:200]}...")
                    elif not product_data["customer_reviews"]:
                        print(f"'customer_reviews' field is empty")
                    else:
                        # Log the number of reviews found
                        print(f"Found {len(product_data['customer_reviews'])} reviews for {product_name}")
                        
                        # Perform sentiment analysis
                        scores = self.analyze_batch_sentiments(product_data["customer_reviews"])
                        print(f"Calculated sentiment scores: {scores}")
                        
                        # Save to cache
                        self.sentiment_cache.set_sentiment(product_name, scores, product_url)
                        
                        return scores
                else:
                    print(f"Scraper returned status code {response.status_code}: {response.text}")
            except requests.RequestException as e:
                print(f"Network error when fetching sentiment data: {str(e)}")
            except Exception as e:
                print(f"Error refreshing sentiment scores: {str(e)}")
                import traceback
                print(traceback.format_exc())
        else:
            print(f"No URL found for {product_name}, skipping direct fetch")
        
        # If not in cache and can't fetch from URL, search Amazon
        try:
            print(f"No direct URL or refresh failed, searching Amazon for {product_name}")
            amazon_data = self.get_amazon_data(product_name)
            
            if amazon_data:
                if "error" in amazon_data:
                    print(f"Error in Amazon search: {amazon_data['error']}")
                elif "customer_reviews" not in amazon_data:
                    print(f"No 'customer_reviews' in Amazon data: {str(amazon_data)[:200]}...")
                elif not amazon_data["customer_reviews"]:
                    print(f"Found 0 reviews from Amazon search")
                else:
                    print(f"Found {len(amazon_data['customer_reviews'])} reviews from Amazon search")
                    
                    # Perform sentiment analysis
                    scores = self.analyze_batch_sentiments(amazon_data["customer_reviews"])
                    print(f"Calculated sentiment scores from Amazon search: {scores}")
                    
                    # Extract URL (for future updates)
                    # In this sample code, it's not clear how to extract the URL, in a real app
                    # you would need to extract the URL from amazon_data
                    product_url = amazon_data.get("product_url", f"https://www.amazon.com/dp/{product_name.replace(' ', '-')}")
                    
                    # Save to cache
                    self.sentiment_cache.set_sentiment(product_name, scores, product_url)
                    
                    return scores
            else:
                print(f"No data returned from Amazon search for {product_name}")
        except Exception as e:
            print(f"Error fetching sentiment data for {product_name}: {str(e)}")
            import traceback
            print(traceback.format_exc())
        
        # If data can't be fetched in any way, use default sentiment scores
        print(f"Using default sentiment scores for {product_name} as all fetch attempts failed")
        self.sentiment_cache.set_sentiment(product_name, self.default_scores)
        return self.default_scores
    
    def process_product(self, product_name: str, category: str, base_confidence: float, usage_freq: Optional[str] = None) -> Optional[ProductRecommendation]:
        """Process a single product and generate recommendation with detailed scores"""
        if pd.notna(product_name):
            try:
                # Get sentiment scores from cache
                sentiment_scores = self.get_product_sentiment(product_name)
                
                # Calculate final score
                final_score = (
                    self.score_weights['knn_weight'] * base_confidence +
                    (1 - self.score_weights['knn_weight']) * sentiment_scores['overall']
                )
                
                return ProductRecommendation(
                    category=category,
                    product_name=product_name,
                    confidence_score=final_score,
                    sentiment_scores=sentiment_scores,
                    usage_frequency=usage_freq
                )
                
            except Exception as e:
                print(f"Error processing product {product_name}: {str(e)}")
                return None
        return None

    def get_recommendations(self, user_features: List[float], wash_frequency: str) -> Dict[str, List[ProductRecommendation]]:
        """Generate recommendations using KNN and sentiment analysis"""
        # Scale features
        scaled_features = self.scaler.transform([user_features])
        
        # Get number of samples in the model
        n_samples = self.knn_model.n_samples_fit_
        
        # Adjust n_neighbors based on available samples
        n_neighbors = min(10, n_samples) if n_samples > 0 else 1
        
        # Update n_neighbors if needed
        if self.knn_model.n_neighbors != n_neighbors:
            print(f"Adjusting n_neighbors from {self.knn_model.n_neighbors} to {n_neighbors} based on available samples")
            self.knn_model.set_params(n_neighbors=n_neighbors)
        
        # Get KNN recommendations
        distances, indices = self.knn_model.kneighbors(scaled_features)
        
        # Get similar users' data
        similar_users_data = []  # This should be your DataFrame with user data
        
        recommendations = {
            RecommendationCategories.SHAMPOO_MONTHLY: [],
            RecommendationCategories.SHAMPOO_WEEKLY: [],
            RecommendationCategories.CONDITIONER: [],
            RecommendationCategories.CONDITIONER_2: [],
            RecommendationCategories.OTHER: []
        }
        
        # Process each similar user's recommendations
        for i, idx in enumerate(indices[0]):
            similar_user = similar_users_data.iloc[idx]
            base_confidence = 1 - (distances[0][i] / np.max(distances))
            
            # Process products
            products_to_process = [
                (similar_user['REC Shampoo MTH'], RecommendationCategories.SHAMPOO_MONTHLY, "Monthly"),
                (similar_user['REC Shampoo WK'], RecommendationCategories.SHAMPOO_WEEKLY, "Weekly"),
                (similar_user['REC Conditioner'], RecommendationCategories.CONDITIONER, None),
                (similar_user['REC Conditioner 2'], RecommendationCategories.CONDITIONER_2, None),
                (similar_user['REC Other'], RecommendationCategories.OTHER, None)
            ]
            
            for product, category, usage_freq in products_to_process:
                if pd.notna(product):
                    rec = self.process_product(product, category, base_confidence, usage_freq)
                    if rec:
                        recommendations[category].append(rec)
        
        # Sort and limit recommendations
        for category in recommendations:
            recommendations[category] = sorted(
                recommendations[category],
                key=lambda x: x.confidence_score,
                reverse=True
            )[:2]
        
        return recommendations

    def store_review(self, review: UserReview):
        """Store new user review"""
        unique_id, _ = self.user_mapping.register_user(
            review.base_name, 
            review.hair_type
        )
        
        now = datetime.now()
        new_review = {
            'base_name': review.base_name,
            'unique_id': unique_id,
            'product_name': review.product_name,
            'hair_type': review.hair_type,
            'rating': review.rating,
            'review_text': review.review_text,
            'purchase_date': review.purchase_date or now,
            'interaction_date': now
        }
        self.user_interactions = pd.concat([
            self.user_interactions,
            pd.DataFrame([new_review])
        ], ignore_index=True)

    def store_user_preference(self, preference: UserPreferences):
        """Store user preferences and calculate condition scores"""
        unique_id = self.user_mapping.generate_user_id(preference.Name)
        
        # Calculate condition scores using the ConditionScores class with proper field mapping
        condition_scores = self.condition_scorer.calculate_condition_scores({
            'issue': preference.Issue.lower() if preference.Issue else "",
            'hair_behaviour': preference.Hair_Behaviour if preference.Hair_Behaviour else "",
            'scalp_feeling': preference.Scalp_Feeling if preference.Scalp_Feeling else "",
            'scalp_flaky': preference.Scalp_Flaky if preference.Scalp_Flaky else "No Not Me",
            'oily_scalp': preference.Oily_Scalp if preference.Oily_Scalp else "",
            'dry_scalp': preference.Dry_Scalp if preference.Dry_Scalp else "",
            'treatments': preference.Treatments if preference.Treatments else "No not me"
        })
        
        now = datetime.now()
        new_data = {
            'Name': preference.Name,
            'Goal': preference.Goal,
            'Issue': preference.Issue,
            'Hair_Type': preference.Hair_Type,
            'Hair_Texture': preference.Hair_Texture,
            'Hair_Behaviour': preference.Hair_Behaviour,
            'Scalp_Feeling': preference.Scalp_Feeling,
            'Wash_Hair': preference.Wash_Hair,
            'Treatments': preference.Treatments,
            'Scalp_Flaky': preference.Scalp_Flaky,
            'Oily_Scalp': preference.Oily_Scalp,
            'Dry_Scalp': preference.Dry_Scalp,
            # Automatically calculated scores
            'DRYNESS': condition_scores['dryness'],
            'DAMAGE': condition_scores['damage'],
            'SENSITIVITY': condition_scores['sensitivity'],
            'SEBUM_Oil': condition_scores['sebum_oil'],
            'DRY_SCALP': condition_scores['dry_scalp_score'],
            'FLAKES': condition_scores['flakes'],
            'timestamp': now.isoformat(),
            'Unique_ID': unique_id
        }

        # Load existing preferences
        preferences_file = os.path.join(self.base_path, "preferences.json")
        preferences_list = []
        if os.path.exists(preferences_file):
            try:
                with open(preferences_file, 'r') as f:
                    preferences_list = json.load(f)
                print(f"Loaded {len(preferences_list)} existing preferences")
            except Exception as e:
                print(f"Error loading preferences: {str(e)}")
                preferences_list = []

        # Update if exists, append if new
        updated = False
        for i, pref in enumerate(preferences_list):
            if pref.get('Unique_ID') == unique_id:
                preferences_list[i] = new_data
                updated = True
                print(f"Updated existing preference for {preference.Name} with ID {unique_id}")
                break

        if not updated:
            preferences_list.append(new_data)
            print(f"Added new preference for {preference.Name} with ID {unique_id}")

        # Save updated preferences
        try:
            with open(preferences_file, 'w') as f:
                json.dump(preferences_list, f, indent=2)
            print(f"Successfully saved preferences. Total preferences: {len(preferences_list)}")
        except Exception as e:
            print(f"Error saving preferences: {str(e)}")
            return None

        if len(preferences_list) >= 5:
            print("You have 5 or more preferences saved. Consider calling /batch_update to update the model.")

        print(f"User preferences saved successfully to preferences.json with ID: {unique_id}")
        return unique_id

    def update_model(self, config: ModelUpdateConfig):
        """Update recommendation models based on new data"""
        if len(self.user_interactions) < config.min_reviews:
            return False, "Insufficient data for model update"
            
        try:
            if config.retrain_knn:
                # Retrain KNN model with new user preferences
                # ... implementation details ...
                pass
                
            return True, "Model updated successfully"
        except Exception as e:
            return False, f"Model update failed: {str(e)}"

    async def batch_update_model(self, min_samples: int = 0):
        """Batch update KNN model with new data"""
        try:
            # Create model versions directory if it doesn't exist
            model_versions_dir = os.path.join(self.base_path, "model_versions")
            if not os.path.exists(model_versions_dir):
                os.makedirs(model_versions_dir)
                print(f"Created directory for model versions: {model_versions_dir}")
            
            # Create model update log file if it doesn't exist
            model_log_path = os.path.join(self.base_path, "model_updates.log")
            
            # Load preferences from JSON file
            preferences_path = os.path.join(self.base_path, "preferences.json")
            if not os.path.exists(preferences_path):
                return False, "preferences.json file does not exist, cannot update model"
                
            # Load existing preferences
            with open(preferences_path, 'r') as f:
                preferences_data = json.load(f)
                
            if len(preferences_data) < min_samples:
                return False, f"Insufficient data for model update. Need {min_samples}, but have {len(preferences_data)} samples."
                
            print(f"Starting batch update of NearestNeighbors model with {len(preferences_data)} samples from preferences.json")
            
            # Convert JSON data to DataFrame
            preferences_df = pd.DataFrame(preferences_data)
            print(f"Loaded preferences with columns: {preferences_df.columns.tolist()}")
            
            # Check for column name consistency
            # Common transformations for column names
            column_mapping = {
                'Base_Name': 'base_name',
                'Unique_ID': 'unique_id',
                'Goal': 'Goal',
                'Issue': 'Issue',
                'Hair Type': 'Hair_Type',
                'Hair Texture': 'Hair_Texture',
                'Hair Behaviour': 'Hair_Behaviour',
                'Scalp Feeling': 'Scalp_Feeling',
                'DRYNESS': 'DRYNESS',
                'DAMAGE': 'DAMAGE',
                'SENSITIVITY': 'SENSITIVITY',
                'SEBUM Oil': 'SEBUM_Oil',
                'DRY SCALP': 'DRY_SCALP',
                'FLAKES': 'FLAKES'
            }
            
            # Standardize column names
            preferences_df = preferences_df.rename(columns={k: v for k, v in column_mapping.items() if k in preferences_df.columns})
            print(f"Standardized column names: {preferences_df.columns.tolist()}")
            
            # Get required columns or use defaults
            required_columns = ['base_name', 'unique_id', 'Goal', 'Issue', 'Hair_Type', 'Hair_Texture', 
                          'Hair_Behaviour', 'Scalp_Feeling', 'DRYNESS', 'DAMAGE', 'SENSITIVITY',
                          'SEBUM_Oil', 'DRY_SCALP', 'FLAKES']
                          
            for col in required_columns:
                if col not in preferences_df.columns:
                    # Add missing column with default values
                    if col in ['DRYNESS', 'DAMAGE', 'SENSITIVITY', 'SEBUM_Oil', 'DRY_SCALP', 'FLAKES']:
                        preferences_df[col] = 0.0  # Default for numeric columns
                    else:
                        preferences_df[col] = "unknown"   # Default for text columns
                    print(f"Added missing column '{col}' with default values")
            
            # Get numerical features
            feature_columns = self.feature_columns
            print(f"Using feature columns from instance: {feature_columns}")

            # Ensure all entries have consistent column names
            print("Normalizing preference keys for all entries in the DataFrame")
            for idx, row in preferences_df.iterrows():
                # Convert row to dictionary, normalize keys, and update DataFrame
                row_dict = row.to_dict()
                normalized_dict = self.normalize_preference_keys(row_dict)
                for key, value in normalized_dict.items():
                    # Burada mevcut eski değer bir Series ise düzeltmek için
                    if key in preferences_df.columns:
                        # Eğer zaten varsa ve tek değerli bir seri değilse
                        # veya pandas değeri ise güvenli bir şekilde atama yapalım
                        preferences_df.at[idx, key] = value
                    else:
                        # Yeni sütun oluşturma gerekiyor
                        preferences_df.loc[idx, key] = value

            # Now ensure all required columns exist in the DataFrame
            for col in feature_columns:
                if col not in preferences_df.columns:
                    preferences_df[col] = 0.0  # Default value for missing numeric columns
                    print(f"Added missing column '{col}' with default values")

            X_numeric = preferences_df[feature_columns].fillna(0.0).values
            print(f"Numeric features shape: {X_numeric.shape}")
            
            # Store feature information for future reference
            feature_info = {
                "feature_columns": feature_columns,
                "feature_count": X_numeric.shape[1],
                "timestamp": datetime.now().isoformat()
            }
            feature_info_path = os.path.join(self.base_path, "feature_info.json")
            with open(feature_info_path, 'w') as f:
                json.dump(feature_info, f, indent=4)
            print(f"Saved feature information to {feature_info_path}")
            
            # Scale numerical features
            X_scaled = self.scaler.fit_transform(X_numeric)
            print(f"Scaled numeric features shape: {X_scaled.shape}")
            
            # Get text features and vectorize, including user identification
            text_columns = ['Goal', 'Issue', 'Hair_Type', 'Hair_Texture', 
                          'Hair_Behaviour', 'Scalp_Feeling']
            
            # Create a combined text feature that includes user name information
            def combine_text_with_user_info(row):
                # Get basic text features
                text_features = []
                for col in text_columns:
                    val = str(row[col]).strip()
                    if val and val.lower() != 'unknown' and val.lower() != 'nan':
                        text_features.append(val)
                
                # Add user identification if available
                unique_id = row.get('unique_id')
                if isinstance(unique_id, (pd.Series, pd.DataFrame)):
                    if not unique_id.empty:
                        unique_id = unique_id.iloc[0]  # İlk değeri al
                if pd.notna(unique_id):
                    text_features.append(str(unique_id))
                
                # Join everything and clean
                text = ' '.join(text_features)
                # Remove special characters and extra spaces
                text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
                text = re.sub(r'\s+', ' ', text)
                return text.strip().lower()
                
            # Apply the function to create the combined text feature
            text_data = preferences_df.apply(combine_text_with_user_info, axis=1)
            
            # Ensure we have some non-empty text data
            non_empty_texts = [t for t in text_data if t.strip()]
            if not non_empty_texts:
                return False, "No valid text data found in preferences"
                
            print(f"Text data sample: {non_empty_texts[0][:100]}")
            
            # Configure vectorizer with minimal preprocessing
            self.vectorizer = TfidfVectorizer(
                stop_words=None,  # Don't remove stop words
                ngram_range=(1, 2),
                max_features=2000,
                min_df=1,
                use_idf=True,
                sublinear_tf=True,
                analyzer='word',
                token_pattern=r'(?u)\b\w+\b'
            )
            
            # Transform text data using vectorizer
            X_text = self.vectorizer.fit_transform(non_empty_texts)
            print(f"Vectorized text features shape: {X_text.shape}")
            
            # Combine features - sparse matrix handling
            if scipy.sparse.issparse(X_text):
                from scipy.sparse import hstack
                X_combined = hstack([X_text, scipy.sparse.csr_matrix(X_scaled)])
                print(f"Using sparse hstack - X_combined shape: {X_combined.shape}")
            else:
                X_combined = np.hstack([X_text.toarray(), X_scaled])
                print(f"Using numpy hstack - X_combined shape: {X_combined.shape}")
                
            print(f"Combined features shape: {X_combined.shape}")
            
            # Update NearestNeighbors model
            from sklearn.neighbors import NearestNeighbors
            
            # First, backup current model if it exists
            model_path = 'C:/Users/HP/Desktop/Project-master/Project-master/V3/nn_model.pkl'
            if os.path.exists(model_path):
                # Copy to backup with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(model_versions_dir, f"nn_model_{timestamp}.pkl")
                import shutil
                shutil.copy2(model_path, backup_path)
                print(f"Backed up existing model to: {backup_path}")
            
            # Create new NearestNeighbors model
            n_neighbors = min(10, len(preferences_data))  # Adjust n_neighbors based on data size
            self.knn_model = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric="cosine",
                algorithm='brute',
                n_jobs=-1
            )
            
            # Fit the model
            self.knn_model.fit(X_combined)
            print("NearestNeighbors model fitting completed")
            
            # Save updated model
            with open(model_path, 'wb') as f:
                pickle.dump(self.knn_model, f)
            print(f"Saved updated NearestNeighbors model to {model_path}")
            
            # Save the scaler and vectorizer
            scaler_path = 'C:/Users/HP/Desktop/Project-master/Project-master/V3/scaler.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"Saved updated scaler to {scaler_path}")
            
            vectorizer_path = 'C:/Users/HP/Desktop/Project-master/Project-master/V3/vectorizer.pkl'
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            print(f"Saved updated vectorizer to {vectorizer_path}")
            
            # Log model update info
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "samples_count": len(preferences_data),
                "feature_count": X_combined.shape[1],
                "feature_columns": feature_columns,
                "numeric_feature_count": X_numeric.shape[1],
                "model_version": backup_path if os.path.exists(model_path) else "first_version",
                "input_columns": preferences_df.columns.tolist()
            }
            
            # Append to log file
            with open(model_log_path, 'a+') as log_file:
                log_file.write(json.dumps(log_entry) + "\n")
            print(f"Logged model update information to {model_log_path}")
            
            # Update the in-memory user_preferences DataFrame
            self.user_preferences = preferences_df
            print("Updated in-memory user_preferences DataFrame")
            
            return True, "NearestNeighbors model updated successfully"
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Model update failed: {str(e)}\n{error_details}")
            return False, f"Model update failed: {str(e)}"

    def save_user_preferences(self, preferences: Dict) -> bool:
        """
        Save user preferences to preferences.json file
        """
        try:
            import os
            import json
            from datetime import datetime
            
            # Normalize preference keys for consistency
            normalized_preferences = self.normalize_preference_keys(preferences)
            
            # Ensure preferences.json exists in the base path
            preferences_file = os.path.join(self.base_path, "preferences.json")
            print(f"Saving preferences to: {preferences_file}")
            
            # Create preferences.json if it doesn't exist
            if not os.path.exists(preferences_file):
                with open(preferences_file, 'w') as f:
                    json.dump([], f)
                    print("Created new preferences.json file")
                existing_preferences = []
            else:
                # Load existing preferences
                try:
                    with open(preferences_file, 'r') as f:
                        existing_preferences = json.load(f)
                        print(f"Loaded {len(existing_preferences)} existing preferences")
                except json.JSONDecodeError:
                    print("Error decoding preferences.json, creating new list")
                    existing_preferences = []

            # Get user name
            user_name = normalized_preferences.get('Name', '')
            if not user_name:
                print("Warning: No user name provided in preferences")
                return False

            # Find existing user and get highest ID
            existing_ids = []
            existing_idx = None
            for idx, pref in enumerate(existing_preferences):
                pref_name = pref.get('Name', '')
                if pref_name == user_name:
                    pref_id = pref.get('Unique_ID', '')
                    if pref_id and '#' in pref_id:
                        try:
                            id_num = int(pref_id.split('#')[1].strip())
                            existing_ids.append(id_num)
                            if existing_idx is None:  # Keep track of first matching preference
                                existing_idx = idx
                        except ValueError:
                            continue

            # Generate or maintain Unique_ID
            if 'Unique_ID' in normalized_preferences:
                # Keep existing ID if provided
                unique_id = normalized_preferences['Unique_ID']
            else:
                # Generate new ID based on highest existing ID + 1
                next_id = max(existing_ids) + 1 if existing_ids else 1
                unique_id = f"{user_name.strip()} #{next_id}"

            # Update preferences with ID and timestamp
            normalized_preferences['Unique_ID'] = unique_id
            normalized_preferences['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Update or append preferences
            if existing_idx is not None:
                # Update existing preference
                existing_preferences[existing_idx] = normalized_preferences
                print(f"Updated existing preference for {user_name} with ID {unique_id}")
            else:
                # Append new preference
                existing_preferences.append(normalized_preferences)
                print(f"Added new preference for {user_name} with ID {unique_id}")

            # Save updated preferences with pretty printing
            with open(preferences_file, 'w') as f:
                json.dump(existing_preferences, f, indent=4)
            
            print(f"Successfully saved preferences. Total preferences: {len(existing_preferences)}")
            
            # Notify if we have enough preferences for model update
            if len(existing_preferences) >= 5:
                print("You have 5 or more preferences saved. Consider calling /batch_update to update the model.")
            
            return True
            
        except Exception as e:
            print(f"Error saving preferences: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False

    def load_feature_info(self):
        """Load feature information from feature_info.json if available"""
        feature_info_path = os.path.join(self.base_path, "feature_info.json")
        if os.path.exists(feature_info_path):
            try:
                with open(feature_info_path, 'r') as f:
                    feature_info = json.load(f)
                    self.feature_columns = feature_info.get("feature_columns", self.feature_columns)
                    print(f"Loaded feature information: {len(self.feature_columns)} features - {self.feature_columns}")
                    return True
            except Exception as e:
                print(f"Could not load feature information: {str(e)}")
        else:
            print("No feature_info.json file found, using default feature columns")
        return False

    def clear_user_preferences(self, user_id: Optional[str] = None, clear_all: bool = False) -> Tuple[bool, str, int, List[str]]:
        """
        Clear user preferences from preferences.json file
        
        Args:
            user_id: Optional user ID to delete a specific user's preferences
            clear_all: If set to true, all preferences will be deleted
            
        Returns:
            Tuple of (success, message, removed_count, removed_ids)
        """
        try:
            preferences_file = os.path.join(self.base_path, "preferences.json")
            if not os.path.exists(preferences_file):
                return False, "No preferences file exists yet", 0, []
                
            # Load existing preferences
            try:
                with open(preferences_file, 'r') as f:
                    preferences = json.load(f)
                    initial_count = len(preferences)
                    print(f"Loaded {initial_count} preferences")
            except json.JSONDecodeError:
                return False, "Error decoding preferences.json file", 0, []
            
            if clear_all:
                # Clear all preferences
                with open(preferences_file, 'w') as f:
                    json.dump([], f)
                
                return True, f"Cleared all preferences ({initial_count} entries removed)", initial_count, []
            else:
                # Remove specific user preferences
                original_length = len(preferences)
                new_preferences = []
                removed_ids = []
                
                for pref in preferences:
                    pref_id = pref.get('Unique_ID') or pref.get('unique_id')
                    
                    # Seri türü kontrolü ve karşılaştırma
                    if isinstance(pref_id, (pd.Series, pd.DataFrame)):
                        # Series ise ilk değeri alalım
                        if not pref_id.empty:
                            pref_id = pref_id.iloc[0]
                        else:
                            pref_id = None
                            
                    # Şimdi güvenli karşılaştırma yapabiliriz
                    if pref_id != user_id:
                        new_preferences.append(pref)
                    else:
                        removed_ids.append(pref_id)
                
                # Save updated preferences
                with open(preferences_file, 'w') as f:
                    json.dump(new_preferences, f, indent=4)
                
                if len(new_preferences) == original_length:
                    return False, f"No preferences found with ID: {user_id}", 0, []
                else:
                    return True, f"Removed {original_length - len(new_preferences)} preferences with ID: {user_id}", original_length - len(new_preferences), removed_ids
        
        except Exception as e:
            print(f"Error clearing preferences: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False, f"Error: {str(e)}", 0, []

    def normalize_preference_keys(self, preferences: Dict) -> Dict:
        """
        Normalize preference keys to handle variations in naming
        
        Args:
            preferences: User preferences dictionary with potentially inconsistent keys
            
        Returns:
            Dictionary with standardized keys
        """
        print(f"Starting key normalization. Input keys: {list(preferences.keys())}")
        normalized = preferences.copy()
        
        # Key mapping for common variations
        key_mapping = {
            # Numeric fields
            'DRYNESS': 'DRYNESS',
            'Dryness': 'DRYNESS',
            'dryness': 'DRYNESS',
            
            'DAMAGE': 'DAMAGE',
            'Damage': 'DAMAGE',
            'damage': 'DAMAGE',
            
            'SENSITIVITY': 'SENSITIVITY',
            'Sensitivity': 'SENSITIVITY',
            'sensitivity': 'SENSITIVITY',
            
            'SEBUM_Oil': 'SEBUM_Oil',
            'SEBUM Oil': 'SEBUM_Oil',
            'Sebum Oil': 'SEBUM_Oil',
            'sebum_oil': 'SEBUM_Oil',
            'sebum oil': 'SEBUM_Oil',
            
            'DRY_SCALP': 'DRY_SCALP',
            'DRY SCALP': 'DRY_SCALP',
            'Dry Scalp': 'DRY_SCALP',
            'dry_scalp': 'DRY_SCALP',
            'dry scalp': 'DRY_SCALP',
            
            'FLAKES': 'FLAKES',
            'Flakes': 'FLAKES',
            'flakes': 'FLAKES',
            
            # Text fields
            'Goal': 'Goal',
            'goal': 'Goal',
            
            'Issue': 'Issue',
            'issue': 'Issue',
            
            'Hair_Type': 'Hair_Type',
            'Hair Type': 'Hair_Type',
            'hair_type': 'Hair_Type',
            'hair type': 'Hair_Type',
            
            'Hair_Texture': 'Hair_Texture',
            'Hair Texture': 'Hair_Texture',
            'hair_texture': 'Hair_Texture',
            'hair texture': 'Hair_Texture',
            
            'Hair_Behaviour': 'Hair_Behaviour',
            'Hair Behaviour': 'Hair_Behaviour',
            'hair_behaviour': 'Hair_Behaviour',
            'hair behaviour': 'Hair_Behaviour',
            
            'Scalp_Feeling': 'Scalp_Feeling',
            'Scalp Feeling': 'Scalp_Feeling',
            'scalp_feeling': 'Scalp_Feeling',
            'scalp feeling': 'Scalp_Feeling',
            
            'Wash_Hair': 'Wash_Hair',
            'Wash Hair': 'Wash_Hair',
            'wash_hair': 'Wash_Hair',
            'wash hair': 'Wash_Hair',
            
            'Treatments': 'Treatments',
            'treatments': 'Treatments',
            
            'Scalp_Flaky': 'Scalp_Flaky',
            'Scalp Flaky': 'Scalp_Flaky',
            'scalp_flaky': 'Scalp_Flaky',
            'scalp flaky': 'Scalp_Flaky',
            
            'Oily_Scalp': 'Oily_Scalp',
            'Oily Scalp': 'Oily_Scalp',
            'oily_scalp': 'Oily_Scalp',
            'oily scalp': 'Oily_Scalp',
            
            'Dry_Scalp': 'Dry_Scalp',
            'Dry Scalp': 'Dry_Scalp',
            'dry_scalp': 'Dry_Scalp',
            'dry scalp': 'Dry_Scalp',
        }
        
        # Process each key in the preferences
        normalized_dict = {}
        for key, value in normalized.items():
            # If key exists in mapping, use standardized version
            mapped_key = key_mapping.get(key, key)
            normalized_dict[mapped_key] = value
            if mapped_key != key:
                print(f"Normalized key: '{key}' → '{mapped_key}'")
        
        # Handle special cases for test data
        # If we have 'SEBUM Oil' but not 'SEBUM_Oil', copy the value
        if 'SEBUM Oil' in normalized_dict and 'SEBUM_Oil' not in normalized_dict:
            # Değer bir pandas Series olabilir, güvenli bir şekilde kopyalayalım
            sebum_oil_value = normalized_dict['SEBUM Oil']
            if hasattr(sebum_oil_value, 'iloc') and len(sebum_oil_value) > 0:
                # Series ise ilk değeri alalım
                sebum_oil_value = sebum_oil_value.iloc[0]
            normalized_dict['SEBUM_Oil'] = sebum_oil_value
            print(f"Copied 'SEBUM Oil' value to 'SEBUM_Oil': {normalized_dict['SEBUM_Oil']}")
        
        # If we have 'DRY SCALP' but not 'DRY_SCALP', copy the value
        if 'DRY SCALP' in normalized_dict and 'DRY_SCALP' not in normalized_dict:
            # Değer bir pandas Series olabilir, güvenli bir şekilde kopyalayalım
            dry_scalp_value = normalized_dict['DRY SCALP']
            if hasattr(dry_scalp_value, 'iloc') and len(dry_scalp_value) > 0:
                # Series ise ilk değeri alalım
                dry_scalp_value = dry_scalp_value.iloc[0]
            normalized_dict['DRY_SCALP'] = dry_scalp_value
            print(f"Copied 'DRY SCALP' value to 'DRY_SCALP': {normalized_dict['DRY_SCALP']}")
        
        # Log any missing required fields
        required_fields = ['DRYNESS', 'DAMAGE', 'SENSITIVITY', 'SEBUM_Oil', 'DRY_SCALP', 'FLAKES']
        missing_fields = []
        for field in required_fields:
            if field not in normalized_dict:
                missing_fields.append(field)
                print(f"Warning: Required field {field} not found in user preferences")
        
        if missing_fields:
            print(f"Missing fields: {missing_fields}")
        else:
            print("All required fields are present after normalization")
        
        print(f"Normalized keys: {list(normalized_dict.keys())}")
        return normalized_dict

    def load_score_weights(self):
        """Load score weights from file or return defaults"""
        weights_file = os.path.join(self.base_path, "score_weights.json")
        if os.path.exists(weights_file):
            try:
                with open(weights_file, 'r') as f:
                    weights = json.load(f)
                print(f"Loaded score weights: KNN={weights['knn_weight']:.2f}, Sentiment={weights['sentiment_weight']:.2f}")
                return weights
            except Exception as e:
                print(f"Error loading score weights: {str(e)}")
        
        # Return default weights
        return {
            'knn_weight': 0.7,
            'sentiment_weight': 0.3
        }

    def load_models(self):
        """Load models and scalers from files"""
        try:
            self.scaler = self.load_pickle('scaler.pkl')
            self.vectorizer = self.load_pickle('vectorizer.pkl')
            self.knn_model = self.load_pickle('nn_model.pkl')
            print("Models and scalers loaded successfully")
        except Exception as e:
            print(f"Error loading models and scalers: {str(e)}")
            import traceback
            print(traceback.format_exc())

# Initialize recommendation engine
engine = RecommendationEngine()

@app.post("/recommend", tags=["recommendations"])
async def get_recommendations(user_preferences: Dict):
    """
    Generate product recommendations based on user preferences using NearestNeighbors model.
    
    This API analyzes user's hair preferences and conditions to recommend personalized hair care products.
    It uses a combination of KNN algorithm for finding similar users and sentiment analysis of product reviews.
    
    Example input:
    {
        "Name": "Clover L",
        "Goal": "Moisture retention",
        "Issue": "Not me",
        "Hair_Type": "Type 4b",
        "Hair_Texture": "Thin or Fine",
        "Hair_Behaviour": "B - Its bouncy and elastic",
        "Scalp_Feeling": "Sensitive",
        "Wash_Hair": "About once a month",
        "Treatments": "No not me",
        "Scalp_Flaky": "No Not Me",
        "Oily_Scalp": "3 - 4 Days",
        "Dry_Scalp": "Within hours"
    }
    """
    try:
        # Import numpy at the beginning
        import numpy as np
        
        # Normalize preference keys to handle variations in field names
        normalized_preferences = engine.normalize_preference_keys(user_preferences)
        print(f"Normalized preference keys. Original keys: {list(user_preferences.keys())}")
        print(f"Normalized keys: {list(normalized_preferences.keys())}")
        
        # Calculate condition scores
        condition_scores = engine.condition_scorer.calculate_condition_scores({
            'issue': normalized_preferences.get('Issue', '').lower(),
            'hair_behaviour': normalized_preferences.get('Hair_Behaviour', ''),
            'scalp_feeling': normalized_preferences.get('Scalp_Feeling', ''),
            'scalp_flaky': normalized_preferences.get('Scalp_Flaky', 'No Not Me'),
            'oily_scalp': normalized_preferences.get('Oily_Scalp', ''),
            'dry_scalp': normalized_preferences.get('Dry_Scalp', ''),
            'treatments': normalized_preferences.get('Treatments', 'No not me')
        })
        
        # Add calculated scores to normalized preferences
        normalized_preferences.update({
            'DRYNESS': condition_scores['dryness'],
            'DAMAGE': condition_scores['damage'],
            'SENSITIVITY': condition_scores['sensitivity'],
            'SEBUM_Oil': condition_scores['sebum_oil'],
            'DRY_SCALP': condition_scores['dry_scalp_score'],
            'FLAKES': condition_scores['flakes']
        })
        
        # Get user name information
        name = normalized_preferences.get('Name', '')
        unique_id = ""
        
        # Save user preferences to preferences.json if a name is provided
        if name:
            # Make a copy to avoid modifying the original
            preferences_to_save = normalized_preferences.copy()
            print(f"Saving preferences for user: {name}")
            
            # Add timestamp
            from datetime import datetime
            preferences_to_save['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Check existing preferences to avoid duplicate IDs
            preferences_file = os.path.join(engine.base_path, "preferences.json")
            next_id = 1
            
            if os.path.exists(preferences_file):
                try:
                    with open(preferences_file, 'r') as f:
                        existing_preferences = json.load(f)
                        
                        # Find existing IDs for this user
                        existing_ids = []
                        for pref in existing_preferences:
                            pref_name = pref.get('Name', '')
                            if pref_name == name:
                                pref_id = pref.get('Unique_ID', '')
                                if pref_id and '#' in pref_id:
                                    try:
                                        id_num = int(pref_id.split('#')[1].strip())
                                        existing_ids.append(id_num)
                                    except ValueError:
                                        pass
                        
                        # Get next available ID
                        if existing_ids:
                            next_id = max(existing_ids) + 1
                            print(f"Found existing IDs for {name}: {existing_ids}, using next ID: {next_id}")
                except Exception as e:
                    print(f"Error reading preferences for ID generation: {str(e)}")
            
            # Generate unique ID with incremented number
            unique_id = f"{name.strip()} #{next_id}"
            preferences_to_save['Unique_ID'] = unique_id
            
            # Save preferences
            save_result = engine.save_user_preferences(preferences_to_save)
            if save_result:
                print(f"User preferences saved successfully to preferences.json with ID: {unique_id}")
            else:
                print(f"Failed to save user preferences")
                return {
                    "status": "error",
                    "message": "Failed to save user preferences"
                }
            
            print(f"User name information: {name} (ID: {unique_id}")
        
        # No need to load CSV data here as the model already has the necessary information
        # Prepare numeric features in the correct order
        numeric_features = []

        # Use the feature columns from the engine instance for consistency
        for col in engine.feature_columns:
            # Get the value from user preferences with appropriate fallback
            numeric_features.append(float(normalized_preferences.get(col, 0)))

        print(f"Preparing {len(numeric_features)} numeric features using columns: {engine.feature_columns}")

        # Scale numeric features
        try:
            scaled_features = engine.scaler.transform([numeric_features])
        except ValueError as e:
            # Check if the error is related to feature count mismatch
            if "features, but StandardScaler is expecting" in str(e):
                # Extract expected feature count from error message
                expected_count = int(str(e).split("expecting")[1].split("features")[0].strip())
                current_count = len(numeric_features)
                print(f"Feature count mismatch detected. Scaler expects {expected_count} features but we have {current_count}.")
                
                # Add placeholder features to match the expected count
                if current_count < expected_count:
                    missing_count = expected_count - current_count
                    print(f"Adding {missing_count} placeholder features to match expected count.")
                    numeric_features.extend([0.0] * missing_count)
                    scaled_features = engine.scaler.transform([numeric_features])
                else:
                    raise ValueError(f"Current feature count {current_count} > expected {expected_count}. Cannot proceed.")
            else:
                # Raise the original error if it's not the feature count issue
                raise
        print(f"Scaled numeric features shape: {scaled_features.shape}")

        # Prepare text features
        text_features = [
            str(normalized_preferences.get('Goal', '')),
            str(normalized_preferences.get('Issue', '')),
            str(normalized_preferences.get('Hair_Type', '')),
            str(normalized_preferences.get('Hair_Texture', '')),
            str(normalized_preferences.get('Hair_Behaviour', '')),
            str(normalized_preferences.get('Scalp_Feeling', '')),
            str(normalized_preferences.get('Wash_Hair', '')),
            str(normalized_preferences.get('Treatments', '')),
            str(normalized_preferences.get('Scalp_Flaky', '')),
            str(normalized_preferences.get('Oily_Scalp', '')),
            str(normalized_preferences.get('Dry_Scalp', ''))
        ]
        
        # Include user name information in text features if available
        if name:
            text_features.append(name)
        if unique_id:
            text_features.append(unique_id)
            
        # Combine and clean text features
        text_combined = ' '.join([str(t).lower() for t in text_features if t])
        text_combined = re.sub(r'[^a-zA-Z0-9\s]', ' ', text_combined)
        text_combined = re.sub(r'\s+', ' ', text_combined)
        text_combined = text_combined.strip()
        
        print(f"Combined text features: {text_combined[:100]}...")

        # Vectorize text features
        text_vector = engine.vectorizer.transform([text_combined])
        print(f"Text vector shape: {text_vector.shape}")

        # Combine features - text vector first, then numeric features
        if scipy.sparse.issparse(text_vector):
            from scipy.sparse import hstack
            X_combined = hstack([text_vector, scipy.sparse.csr_matrix(scaled_features)])
            print(f"Using sparse hstack - X_combined shape: {X_combined.shape}")
        else:
            X_combined = np.hstack([text_vector.toarray(), scaled_features])
            print(f"Using numpy hstack - X_combined shape: {X_combined.shape}")

        # Get number of samples in the model
        n_samples = engine.knn_model.n_samples_fit_
        print(f"Number of samples in KNN model: {n_samples}")
        
        # Adjust n_neighbors based on available samples
        n_neighbors = min(10, n_samples) if n_samples > 0 else 1
        
        # Update n_neighbors if needed
        if engine.knn_model.n_neighbors != n_neighbors:
            print(f"Adjusting n_neighbors from {engine.knn_model.n_neighbors} to {n_neighbors}")
            engine.knn_model.set_params(n_neighbors=n_neighbors)

        # Get KNN recommendations
        distances, indices = engine.knn_model.kneighbors(X_combined)
        
        # Load the data source for recommendations (since we removed the CSV loading earlier)
        try:
            # Try to get data from trained model or preferences file
            if hasattr(engine, 'user_preferences') and isinstance(engine.user_preferences, pd.DataFrame) and not engine.user_preferences.empty:
                df = engine.user_preferences
                print(f"Using in-memory user_preferences data with {len(df)} records")
            else:
                # Load data from preferences.json as fallback
                preferences_path = os.path.join(engine.base_path, "preferences.json")
                if os.path.exists(preferences_path):
                    with open(preferences_path, 'r') as f:
                        preferences_data = json.load(f)
                        df = pd.DataFrame(preferences_data)
                        print(f"Loaded {len(df)} preferences from {preferences_path}")
                else:
                    # Last resort - only load CSV if we have no other choice
                    csv_path = 'C:/Users/HP/Desktop/Project-master/Project-master/V3/synthetic_data_indexed.csv'
                    df = pd.read_csv(csv_path)
                    print(f"Loaded {len(df)} records from CSV: {csv_path} (fallback)")
        except Exception as data_error:
            print(f"Error accessing model data: {str(data_error)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to access model data: {str(data_error)}"
            )
            
        # Rest of your existing recommendation logic...
        # ... [Rest of the recommendation processing code remains the same]

        # Extract condition scores
        condition_scores = {
            "DRYNESS": normalized_preferences.get('DRYNESS', 0),
            "DAMAGE": normalized_preferences.get('DAMAGE', 0),
            "SENSITIVITY": normalized_preferences.get('SENSITIVITY', 0),
            "SEBUM Oil": normalized_preferences.get('SEBUM_Oil', 0),
            "DRY SCALP": normalized_preferences.get('DRY_SCALP', 0),
            "FLAKES": normalized_preferences.get('FLAKES', 0)
        }
        
        # Determine primary condition
        primary_condition = max(condition_scores, key=condition_scores.get)
        print(f"Primary condition: {primary_condition} (Score: {condition_scores[primary_condition]:.4f})")
        
        # Determine condition category
        condition_to_category = {
            'DRYNESS': 'Moisture',
            'DAMAGE': 'Damage',
            'DRY SCALP': 'Scalp',
            'FLAKES': 'Scalp',
            'SEBUM Oil': 'Scalp',
            'SENSITIVITY': 'Sensitivity'
        }
        
        primary_category = condition_to_category.get(primary_condition, 'Other')
        
        # Get recommendations from nearest neighbors
        recommendations = {}
        
        # Mapping dictionary
        rec_mappings = {
            'REC Shampoo MTH': ['REC Shampoo MTH'],
            'REC Shampoo WK': ['REC Shampoo WK'],
            'REC Conditioner': ['REC Conditioner'],
            'REC Conditioner 2': ['REC Conditioner 2'],
            'REC Other ': ['REC Other ', 'REC Other']
        }
        
        # Store categorized products
        categorized_products = {
            'Shampoo': [],
            'Conditioner': [],
            'Other': []
        }
        
        # Examine the nearest neighbors
        for i, idx in enumerate(indices[0]):
            if idx >= len(df):
                print(f"Warning: Index {idx} out of bounds for DataFrame with {len(df)} rows")
                continue
                
            row = df.iloc[idx]
            similarity_score = 1 - distances[0][i]
            
            print(f"Neighbor {i+1} - Similarity: {similarity_score:.4f}")

            # Process recommendations for each category
            for rec_key, rec_variants in rec_mappings.items():
                for variant in rec_variants:
                    if variant in row:
                        product_name = row[variant]
                        
                        if pd.isna(product_name) or not isinstance(product_name, str) or not product_name.strip():
                            continue
                            
                        product_lower = product_name.lower()
                        category = None
                        
                        if any(term in product_lower for term in ['shampoo', 'wash', 'cleanser']):
                            category = 'Shampoo'
                        elif any(term in product_lower for term in ['conditioner', 'condition']):
                            category = 'Conditioner'
                        else:
                            category = 'Other'
                            
                        categorized_products[category].append({
                            'product': product_name,
                            'similarity': similarity_score,
                            'source': variant
                        })
                        break

        # Prioritize based on category
        if primary_category == 'Moisture':
            priority_categories = ['Conditioner', 'Shampoo', 'Other']
        elif primary_category == 'Damage':
            priority_categories = ['Conditioner', 'Other', 'Shampoo']
        elif primary_category == 'Scalp':
            priority_categories = ['Shampoo', 'Other', 'Conditioner']
        elif primary_category == 'Sensitivity':
            priority_categories = ['Shampoo', 'Conditioner', 'Other']
        else:
            priority_categories = ['Shampoo', 'Conditioner', 'Other']
        
        # Final recommendations
        final_recommendations = {}
        
        for category in priority_categories:
            sorted_products = sorted(
                categorized_products[category], 
                key=lambda x: x['similarity'], 
                reverse=True
            )
            
            top_products = sorted_products[:2]
            
            if top_products:
                category_key = f"{category}"
                final_recommendations[category_key] = []
                
                for product_info in top_products:
                    product_name = product_info['product']
                    sentiment_scores = engine.get_product_sentiment(product_name)
                    
                    final_score = (
                        engine.score_weights['knn_weight'] * product_info['similarity'] +
                        (1 - engine.score_weights['knn_weight']) * sentiment_scores['overall']
                    )
                    
                    final_recommendations[category_key].append({
                        'product': product_name,
                        'similarity_score': product_info['similarity'],
                        'sentiment_scores': sentiment_scores,
                        'final_score': final_score,
                        'source': product_info['source']
                    })
        
        # Return results
        response = {
            "recommendations": final_recommendations,
            "condition_scores": condition_scores,
            "primary_condition": primary_condition,
            "primary_category": primary_category
        }
        
        if name:
            response["user"] = {
                "base_name": name,
                "unique_id": unique_id
            }
        
        return response

    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_preferences", tags=["users"])
async def update_preferences(preference: UserPreferences):
    """
    Update user preferences and save to preferences.json
    
    This endpoint allows you to store or update a single user's preferences in the system.
    The preferences are used by the recommendation algorithm to find personalized product suggestions.
    
    Example input:
    {
        "Name": "Clover L",
        "Goal": "Moisture retention",
        "Issue": "Not me",
        "Hair_Type": "Type 4b",
        "Hair_Texture": "Thin or Fine",
        "Hair_Behaviour": "B - Its bouncy and elastic",
        "Scalp_Feeling": "Sensitive",
        "Wash_Hair": "About once a month",
        "Treatments": "No not me",
        "Scalp_Flaky": "No Not Me",
        "Oily_Scalp": "3 - 4 Days",
        "Dry_Scalp": "Within hours"
    }
    """
    try:
        # Store user preferences and calculate condition scores
        unique_id = engine.store_user_preference(preference)
        if unique_id:
            return {
                "status": "success", 
                "message": "Preferences updated successfully",
                "unique_id": unique_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update preferences")
    except Exception as e:
        print(f"Error in update_preferences: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_update", tags=["model"])
async def batch_update(
    background_tasks: BackgroundTasks,
    min_samples: int = Query(5, description="Minimum number of new or modified samples required to trigger update"),
    force_update: bool = Query(False, description="Force model update regardless of sample count"),
    check_only: bool = Query(False, description="Only check if update is needed without performing it")
):
    """
    Trigger batch update of the recommendation model with safety checks
    
    This endpoint will analyze the current state and only update the model if necessary:
    1. Checks the number of new and modified samples in preferences.json
    2. Compares with existing model data using content hashing
    3. Updates only if there are sufficient changes or if forced
    
    Parameters:
    - min_samples: Minimum number of new or modified samples required to trigger update (default: 5)
    - force_update: Force model update regardless of sample count
    - check_only: Only check if update is needed without performing it
    
    The update will only proceed if:
    - There are at least min_samples new or modified samples in preferences.json
    - OR force_update is True
    """
    try:
        # Get current model information
        model_info = {
            "model_exists": False,
            "model_samples": 0,
            "model_last_updated": None,
            "model_sample_hashes": set()  # Store hashes of samples in model
        }
        
        # Check existing model and load sample hashes if available
        model_path = os.path.join(engine.base_path, "nn_model.pkl")
        model_hashes_path = os.path.join(engine.base_path, "model_sample_hashes.json")
        
        if os.path.exists(model_path):
            model_info["model_exists"] = True
            if hasattr(engine.knn_model, 'n_samples_fit_'):
                model_info["model_samples"] = engine.knn_model.n_samples_fit_
            
            # Get model last modified time
            model_info["model_last_updated"] = datetime.fromtimestamp(
                os.path.getmtime(model_path)
            ).isoformat()
            
            # Load sample hashes if available
            if os.path.exists(model_hashes_path):
                try:
                    with open(model_hashes_path, 'r') as f:
                        model_info["model_sample_hashes"] = set(json.load(f))
                except Exception as e:
                    print(f"Error loading model hashes: {str(e)}")
        
        # Check preferences.json
        preferences_path = os.path.join(engine.base_path, "preferences.json")
        preferences_info = {
            "exists": False,
            "samples": 0,
            "last_updated": None,
            "new_samples": 0,
            "modified_samples": 0,
            "unchanged_samples": 0
        }
        
        if os.path.exists(preferences_path):
            preferences_info["exists"] = True
            try:
                with open(preferences_path, 'r') as f:
                    preferences_data = json.load(f)
                    preferences_info["samples"] = len(preferences_data)
                    preferences_info["last_updated"] = datetime.fromtimestamp(
                        os.path.getmtime(preferences_path)
                    ).isoformat()
                    
                    # Calculate hashes for current samples
                    current_hashes = set()
                    for sample in preferences_data:
                        # Create a stable hash of the sample's content
                        sample_str = json.dumps(sample, sort_keys=True)
                        sample_hash = md5(sample_str.encode()).hexdigest()
                        current_hashes.add(sample_hash)
                        
                        # Check if this is a new or modified sample
                        if sample_hash not in model_info["model_sample_hashes"]:
                            if len(model_info["model_sample_hashes"]) == 0:
                                preferences_info["new_samples"] += 1
                            else:
                                # If we have model hashes but this one is new, it's modified
                                preferences_info["modified_samples"] += 1
                        else:
                            preferences_info["unchanged_samples"] += 1
                    
                    # Save current hashes for next comparison
                    with open(model_hashes_path, 'w') as f:
                        json.dump(list(current_hashes), f)
                    
            except Exception as e:
                print(f"Error reading preferences.json: {str(e)}")
                preferences_info["error"] = str(e)
        
        # Calculate total changes
        total_changes = preferences_info["new_samples"] + preferences_info["modified_samples"]
        
        # Determine if update is needed
        update_needed = (
            force_update or
            not model_info["model_exists"] or
            total_changes >= min_samples
        )
        
        # Prepare status response
        status = {
            "model_info": model_info,
            "preferences_info": preferences_info,
            "total_changes": total_changes,
            "update_needed": update_needed,
            "min_samples_required": min_samples,
            "force_update": force_update
        }
        
        # If only checking status, return here
        if check_only:
            return {
                "status": "success",
                "message": "Update check completed",
                "update_info": status
            }
        
        # If update is needed and not just checking
        if update_needed:
            if force_update:
                print("Forcing model update")
            else:
                print(f"Update needed: {total_changes} changes found ({preferences_info['new_samples']} new, {preferences_info['modified_samples']} modified)")
            
            # Perform the update
            result, message = await engine.batch_update_model(min_samples=0 if force_update else min_samples)
            
            return {
                "status": "success" if result else "error",
                "message": message,
                "update_info": status
            }
        else:
            return {
                "status": "skipped",
                "message": f"Update not needed. Only {total_changes} changes (minimum required: {min_samples})",
                "update_info": status
            }
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Batch update error: {str(e)}\n{error_details}")
        return {
            "status": "error",
            "message": f"Batch update failed: {str(e)}",
            "error_details": error_details
        }

@app.post("/refresh-sentiments", tags=["model"])
async def refresh_sentiments(background_tasks: BackgroundTasks):
    """
    Periodically refresh sentiment scores for all products.
    This endpoint can be called manually or scheduled as a cron job for automatic updates.
    
    The system will identify products that need refreshing based on their last update time
    and queue them for background processing to avoid blocking the main thread.
    """
    try:
        # Find products that need refreshing
        products_to_refresh = [
            product_name for product_name in engine.sentiment_cache.sentiments
            if engine.sentiment_cache.needs_refresh(product_name)
        ]
        
        # Start background refresh task
        background_tasks.add_task(refresh_product_sentiments, products_to_refresh)
        
        return {
            "message": f"Started background refresh for {len(products_to_refresh)} products",
            "products": products_to_refresh
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def refresh_product_sentiments(product_names):
    """Refresh sentiment scores for products"""
    for product_name in product_names:
        try:
            engine.get_product_sentiment(product_name)  # This function will handle refreshing
            await asyncio.sleep(2)  # Sleep to avoid overloading the API
        except Exception as e:
            print(f"Error refreshing sentiment for {product_name}: {str(e)}")

@app.post("/set-product-sentiment", tags=["model"])
async def set_product_sentiment(
    product_name: str, 
    likability: float = 0.65,
    effectiveness: float = 0.7,
    value_for_money: float = 0.6,
    ingredient_quality: float = 0.65,
    ease_of_use: float = 0.7
):
    """
    Manually set sentiment scores for a product.
    This can be used for products not found on Amazon or for custom products.
    
    You can specify individual scores for different aspects, which will be combined
    with appropriate weights to calculate an overall sentiment score.
    """
    try:
        # Create sentiment scores
        sentiment_scores = {
            'likability': likability,
            'effectiveness': effectiveness,
            'value_for_money': value_for_money,
            'ingredient_quality': ingredient_quality,
            'ease_of_use': ease_of_use,
            'overall': (likability * 0.3 + effectiveness * 0.3 + value_for_money * 0.2 +
                        ingredient_quality * 0.1 + ease_of_use * 0.1)
        }
        
        # Save to cache
        engine.sentiment_cache.set_sentiment(product_name, sentiment_scores)
        
        return {
            "message": f"Set sentiment scores for {product_name}",
            "sentiment_scores": sentiment_scores
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Load product data
try:
    product_file = "product_names.json"
    if os.path.exists(product_file):
        with open(product_file, 'r') as f:
            product_data = json.load(f)
            all_products = product_data.get("all_products", [])
            print(f"Loaded {len(all_products)} products from {product_file}")
    else:
        print(f"Warning: {product_file} not found. Run extract_products.py first.")
        all_products = []
        product_data = {"all_products": [], "product_by_category": {}}
except Exception as e:
    print(f"Error loading product data: {str(e)}")
    all_products = []
    product_data = {"all_products": [], "product_by_category": {}}

# Kategorilere ayırıp tüm ürünleri dahil edelim
categories = {
    "Shampoo": [p for p in all_products if "Shampoo" in p or "shampoo" in p],
    "Conditioner": [p for p in all_products if "Conditioner" in p or "conditioner" in p],
    "Other": [p for p in all_products if "Conditioner" not in p and "conditioner" not in p 
                                and "Shampoo" not in p and "shampoo" not in p]
}

# Tüm ürünleri birleştirip alfebetik sırayla listeleyelim
simplified_products = []
for category, products in categories.items():
    simplified_products.extend(products)
simplified_products = sorted(list(set(simplified_products)))

# Hiç ürün yoksa örnek ekleyelim (test için)
if not simplified_products:
    simplified_products = ["TestShampoo", "TestConditioner", "TestOther"]

@app.post("/add-product-url-swagger", tags=["model"])
async def add_product_url_swagger(
    product_name: str = Query(
        ..., 
        description="Product name",
        enum=simplified_products
    ),
    product_url: str = Query(
        ..., 
        description="Amazon product URL",
        example="https://www.amazon.com/dp/XXXXX"
    ),
    force_refresh: bool = Query(
        True, 
        description="Recalculate sentiment scores when URL is added"
    )
):
    """
    Add URL for a product by selecting the product name from a dropdown menu.
    
    This API allows you to associate an Amazon product URL with a product in our database.
    The system will use this URL to fetch customer reviews and calculate sentiment scores.
    """
    try:
        # Save the URL first
        engine.sentiment_cache.set_url(product_name, product_url)
        
        print(f"Adding URL for {product_name}: {product_url}")
        
        # Immediately refresh sentiment scores from new URL (with force_refresh)
        sentiment_scores = engine.get_product_sentiment(product_name, force_refresh=force_refresh)
        
        return {
            "message": f"Added URL for {product_name}",
            "url": product_url,
            "sentiment_scores": sentiment_scores,
            "refreshed": force_refresh
        }
    except Exception as e:
        import traceback
        print(f"Error adding product URL via swagger: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoints
class ProductSelection(BaseModel):
    product_name: str = Field(..., description="Product name")
    
    @validator('product_name')
    def product_must_exist(cls, v):
        if v not in all_products:
            raise ValueError(f"Product not found: {v}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "product_name": all_products[0] if all_products else "ExampleProduct"
            }
        }

@app.post("/add-product-url-dropdown", tags=["model"], deprecated=True)
async def add_product_url_dropdown(
    selection: ProductSelection,
    product_url: str,
    force_refresh: bool = Query(True, description="Recalculate sentiment scores when URL is added")
):
    """
    [DEPRECATED - Use /add-product-url-swagger instead] 
    Add URL for a product by selecting the product name from a model.
    """
    try:
        product_name = selection.product_name
        
        # Save the URL first
        engine.sentiment_cache.set_url(product_name, product_url)
        
        print(f"Adding URL for {product_name}: {product_url}")
        
        # Immediately refresh sentiment scores from new URL (with force_refresh)
        sentiment_scores = engine.get_product_sentiment(product_name, force_refresh=force_refresh)
        
        return {
            "message": f"Added URL for {product_name}",
            "url": product_url,
            "sentiment_scores": sentiment_scores
        }
    except Exception as e:
        import traceback
        print(f"Error adding product URL via dropdown: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-product-url", tags=["model"])
async def add_product_url(
    product_name: str,  # Take product name as string
    product_url: str,
    force_refresh: bool = Query(True, description="Recalculate sentiment scores when URL is added")
):
    """
    Add URL for a product by entering the product name. Use for UIs outside of Swagger.
    
    * **product_name**: Product name (use /products endpoint to see all products)
    * **product_url**: Amazon product URL
    * **force_refresh**: If True, clears existing cache data and recalculates
    """
    try:
        # Validate product name
        if product_name not in all_products:
            # Find closest match
            closest_match = None
            highest_similarity = 0
            
            for p in all_products:
                similarity = SequenceMatcher(None, product_name.lower(), p.lower()).ratio()
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    closest_match = p
            
            if highest_similarity > 0.8:  # If similarity is higher than 0.8
                product_name = closest_match
                print(f"Using closest match: {product_name}")
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Product name not found. Closest match: {closest_match if closest_match else 'None'}"
                )
        
        # Save the URL first
        engine.sentiment_cache.set_url(product_name, product_url)
        
        print(f"Adding URL for {product_name}: {product_url}")
        
        # Immediately refresh sentiment scores from new URL (with force_refresh)
        sentiment_scores = engine.get_product_sentiment(product_name, force_refresh=force_refresh)
        
        return {
            "message": f"Added URL for {product_name}",
            "url": product_url,
            "sentiment_scores": sentiment_scores,
            "refreshed": force_refresh
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        print(f"Error adding product URL: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/products", tags=["model"])
async def get_products():
    """
    Return a list of all available products.
    You can use this list to create a dropdown menu in your UI.
    
    This endpoint returns both products from the database and those that have been
    added to the sentiment cache, organized by categories.
    """
    try:
        # Include products from cache
        cached_products = list(engine.sentiment_cache.sentiments.keys())
        
        # Combine cached products and products from JSON
        all_unique_products = sorted(list(set(all_products + cached_products)))
        
        # Use category information from JSON
        product_categories = product_data.get("product_by_category", {})
        
        # Add custom products from cache
        product_categories["Custom Products"] = [p for p in cached_products if p not in all_products]
        
        return {
            "total_products": len(all_unique_products),
            "categories": product_categories,
            "products": all_unique_products
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ProductUrlMapping(BaseModel):
    product_name: str
    product_url: str

@app.post("/add-multiple-product-urls", tags=["model"])
async def add_multiple_product_urls(mappings: List[ProductUrlMapping]):
    """
    Add multiple product-URL mappings at once.
    
    This batch operation allows you to associate multiple products with their respective
    Amazon URLs in a single API call, which is more efficient for bulk operations.
    Each mapping includes a product name and its corresponding URL.
    """
    results = []
    for mapping in mappings:
        try:
            engine.sentiment_cache.set_url(mapping.product_name, mapping.product_url)
            results.append({
                "product_name": mapping.product_name,
                "url": mapping.product_url,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "product_name": mapping.product_name,
                "url": mapping.product_url,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "message": f"Processed {len(mappings)} product-URL mappings",
        "results": results
    }

@app.post("/clear-product-sentiment", tags=["model"])
async def clear_product_sentiment(product_name: str):
    """
    Clear sentiment scores for a specific product in the cache.
    This will force the system to recalculate sentiment scores on the next request.
    
    The product's URL is preserved, only the sentiment data is cleared.
    """
    try:
        product_url = engine.sentiment_cache.clear_sentiment(product_name)
        
        if product_url:
            return {
                "message": f"Cleared sentiment data for {product_name}",
                "product_url": product_url,
                "status": "success" 
            }
        else:
            return {
                "message": f"No sentiment data found for {product_name}",
                "status": "warning"
            }
    except Exception as e:
        import traceback
        print(f"Error clearing sentiment data: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-all-sentiments", tags=["model"])
async def clear_all_sentiments():
    """
    Clear sentiment scores for all products in the cache.
    This is useful for quickly resetting the cache without losing URL information.
    
    Only sentiment scores are deleted, all product URLs are preserved.
    """
    try:
        engine.sentiment_cache.clear_all_sentiments()
        return {
            "message": "All sentiment data has been cleared",
            "status": "success"
        }
    except Exception as e:
        import traceback
        print(f"Error clearing all sentiment data: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_user_names", tags=["users"])
async def list_user_names():
    """
    Get a list of all user names from preferences.json
    This endpoint returns all available user names that can be used for name-based recommendations.
    """
    try:
        preferences_path = os.path.join(engine.base_path, "preferences.json")
        if not os.path.exists(preferences_path):
            return {
                "status": "warning",
                "message": "No preferences file exists yet",
                "names": []
            }
            
        with open(preferences_path, 'r') as f:
            preferences_data = json.load(f)
            
        # Extract unique names from preferences
        names = set()
        for pref in preferences_data:
            name = pref.get('Name') or pref.get('base_name')
            if name:
                names.add(name)
        
        return {
            "status": "success",
            "message": f"Found {len(names)} unique user names",
            "names": sorted(list(names))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class NameBasedRecommendationRequest(BaseModel):
    Name: str = Field(..., description="User name to search for recommendations")
    
    class Config:
        schema_extra = {
            "example": {
                "Name": "Clover L"
            }
        }

@app.post("/recommend_by_name", tags=["recommendations"])
async def get_recommendations_by_name(
    request: NameBasedRecommendationRequest = Body(
        ...,
        examples={
            "normal": {
                "summary": "Normal example",
                "description": "A normal example with a user name",
                "value": {"Name": "Clover L"}
            },
            "partial": {
                "summary": "Partial name example",
                "description": "An example with partial name match",
                "value": {"Name": "Clover"}
            }
        }
    )
):
    """
    Generate product recommendations based on user name matching.
    This endpoint finds similar users by name and returns their recommendations.
    
    The name matching is case-insensitive and supports partial matches.
    For example, searching for "Clover" will match "Clover L" and "Clover Smith".
    
    You can use the /list_user_names endpoint to see all available user names.
    """
    try:
        base_name = request.Name
        
        if not base_name:
            raise HTTPException(
                status_code=400, 
                detail="User name (Name) is required for name-based recommendations"
            )
            
        print(f"Searching for recommendations for user: {base_name}")
        
        # Clean name and generate search patterns
        clean_base_name = base_name.strip().lower()
        
        # Load user preferences data
        try:
            # Try to load from the preferences path first
            preferences_path = os.path.join(engine.base_path, "preferences.json")
            if os.path.exists(preferences_path):
                with open(preferences_path, 'r') as f:
                    preferences_data = json.load(f)
                    preferences_df = pd.DataFrame(preferences_data)
                    print(f"Loaded {len(preferences_df)} user preferences from {preferences_path}")
            else:
                # Use the trained model's data instead of loading CSV again
                print("No preferences.json found. Using model's existing data.")
                return {
                    "message": f"No user data available for matching with '{base_name}'",
                    "recommendations": {}
                }
        except Exception as e:
            print(f"Error loading preferences data: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to load user preferences data: {str(e)}"
            )
        
        # Add the Base_Name column check
        has_base_name = False
        name_column = None
        
        # Check for base name column - could be Base_Name, base_name, or NAME
        possible_name_columns = ['Name', 'Base_Name', 'base_name', 'NAME', 'name']
        for col in possible_name_columns:
            if col in preferences_df.columns:
                name_column = col
                has_base_name = True
                break
                
        if not has_base_name:
            print("No name column found in data. Available columns:", preferences_df.columns.tolist())
            raise HTTPException(
                status_code=500,
                detail="User name column not found in the dataset"
            )
        
        # Fill NA/NaN values with empty string to avoid comparison errors
        preferences_df[name_column] = preferences_df[name_column].fillna('')
        
        # Check if we have the exact name match
        matching_users = preferences_df[preferences_df[name_column].str.lower().fillna('') == clean_base_name]
        
        # If no exact match, try partial matching
        if len(matching_users) == 0:
            print(f"No exact match for '{clean_base_name}', trying partial matches")
            # Use str.contains with na=False to handle NA/NaN values
            matching_users = preferences_df[preferences_df[name_column].str.lower().str.contains(clean_base_name, na=False)]
        
        # Still no matches, return appropriate message
        if len(matching_users) == 0:
            print(f"No matching users found for '{clean_base_name}'")
            return {
                "message": f"No matching users found for '{base_name}'",
                "recommendations": {}
            }
            
        print(f"Found {len(matching_users)} matching users")
        
        # Create user preferences from the first matching user's data
        best_match = matching_users.iloc[0]
        user_preferences = {}
        
        # Map standard fields
        condition_fields = ["DRYNESS", "DAMAGE", "SENSITIVITY", "SEBUM_Oil", "DRY_SCALP", "FLAKES"]
        text_fields = ["Goal", "Issue", "Hair_Type", "Hair_Texture", "Hair_Behaviour", "Scalp_Feeling"]
        
        # Add condition scores
        for field in condition_fields:
            if field in best_match and pd.notna(best_match[field]):
                user_preferences[field] = float(best_match[field])  # Convert to float
            else:
                user_preferences[field] = 0.0  # Default value
                
        # Add text fields
        for field in text_fields:
            if field in best_match and pd.notna(best_match[field]):
                user_preferences[field] = str(best_match[field])  # Convert to string
            else:
                user_preferences[field] = ""  # Default value
        
        # Add name
        user_preferences['Name'] = str(best_match[name_column])
        
        # Forward to the standard recommendation endpoint
        recommendations = await get_recommendations(user_preferences)
        
        # Add matching user info to response
        recommendations["matched_user"] = {
            "name": str(best_match[name_column]),
            "unique_id": str(best_match.get("Unique_ID", "Unknown"))
        }
        
        return recommendations
        
    except Exception as e:
        print(f"Error in get_recommendations_by_name: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_update_preferences", tags=["users"])
async def batch_update_preferences(preferences: List[UserPreferences]):
    """
    Batch update multiple user preferences and save to preferences.json
    
    This endpoint allows you to submit multiple user preference objects in a single request,
    which is more efficient for bulk operations. Each preference object will be processed
    and stored individually in the preferences database.
    """
    try:
        success_count = 0
        failed_count = 0
        
        for preference in preferences:
            try:
                # Convert preference to dict
                preference_dict = preference.dict()
                # Fix field naming to match expected format
                if "Hair_Type" in preference_dict:
                    preference_dict["Hair_Type"] = preference_dict.pop("Hair_Type")
                if "Hair_Texture" in preference_dict:
                    preference_dict["Hair_Texture"] = preference_dict.pop("Hair_Texture")
                if "Hair_Behaviour" in preference_dict:
                    preference_dict["Hair_Behaviour"] = preference_dict.pop("Hair_Behaviour")
                if "Scalp_Feeling" in preference_dict:
                    preference_dict["Scalp_Feeling"] = preference_dict.pop("Scalp_Feeling")
                if "SEBUM_Oil" in preference_dict:
                    preference_dict["SEBUM_Oil"] = preference_dict.pop("SEBUM_Oil")
                if "DRY_SCALP" in preference_dict:
                    preference_dict["DRY_SCALP"] = preference_dict.pop("DRY_SCALP")
                
                # Save user preferences
                result = engine.save_user_preferences(preference_dict)
                if result:
                    success_count += 1
                else:
                    failed_count += 1
            except Exception as item_error:
                print(f"Error processing preference: {str(item_error)}")
                failed_count += 1
        
        return {
            "status": "success", 
            "message": f"Processed {len(preferences)} preferences",
            "success_count": success_count,
            "failed_count": failed_count
        }
    except Exception as e:
        print(f"Error in batch_update_preferences: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# CSV'den toplu yükleme için yeni bir endpoint
@app.post("/import_preferences_from_csv", tags=["users"])
async def import_preferences_from_csv(file: UploadFile = File(...)):
    """
    Bulk import user preferences from a CSV file and save to preferences.json
    
    This endpoint allows you to upload a CSV file containing user preferences,
    which will be processed and stored in the system's preference database.
    The file should contain columns matching the expected user preference fields.
    """
    try:
        # Read CSV file
        contents = await file.read()
        try:
            # Save to temporary file
            with open("temp_preferences.csv", "wb") as f:
                f.write(contents)
            
            # Read CSV with pandas
            df = pd.read_csv("temp_preferences.csv")
            print(f"Loaded CSV with {len(df)} rows and columns: {df.columns.tolist()}")
            
            # Match with expected column names
            column_mapping = {
                'Base_Name': 'base_name',
                'Goal': 'Goal',
                'Issue': 'Issue',
                'Hair Type': 'Hair_Type',
                'Hair Texture': 'Hair_Texture',
                'Hair Behaviour': 'Hair_Behaviour',
                'Scalp Feeling': 'Scalp_Feeling',
                'DRYNESS': 'DRYNESS',
                'DAMAGE': 'DAMAGE',
                'SENSITIVITY': 'SENSITIVITY',
                'SEBUM Oil': 'SEBUM_Oil',
                'DRY SCALP': 'DRY_SCALP',
                'FLAKES': 'FLAKES'
            }
            
            # Sütun isimlerini standartlaştır
            df = df.rename(columns={v: k for k, v in column_mapping.items() if v in df.columns})
            
            # Her satırı bir UserPreferences nesnesine dönüştür
            preferences = []
            for _, row in df.iterrows():
                try:
                    # NaN değerleri filtrele
                    preference_dict = {k: v for k, v in row.items() if pd.notna(v)}
                    
                    # UserPreferences nesnesine dönüştür
                    preference = UserPreferences(**preference_dict)
                    preferences.append(preference)
                except Exception as row_error:
                    print(f"Error converting row to preference: {str(row_error)}")
            
            # Toplu güncelleme yap
            return await batch_update_preferences(preferences)
            
        finally:
            # Geçici dosyayı temizle
            import os
            if os.path.exists("temp_preferences.csv"):
                os.remove("temp_preferences.csv")
    
    except Exception as e:
        print(f"Error importing preferences from CSV: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-preferences", tags=["users"])
async def clear_preferences(
    user_id: Optional[str] = None,
    clear_all: bool = False
):
    """
    Clear user preferences from preferences.json file
    
    This endpoint allows you to remove specific user preferences or clear all preferences.
    
    Parameters:
    - user_id: Optional user ID to delete a specific user's preferences
    - clear_all: If set to true, all preferences will be deleted
    
    At least one parameter must be provided. If both are provided, clear_all takes precedence.
    """
    try:
        if not user_id and not clear_all:
            raise HTTPException(
                status_code=400,
                detail="Either user_id or clear_all must be provided"
            )
            
        success, message, removed_count, removed_ids = engine.clear_user_preferences(user_id, clear_all)
        
        if success:
            return {
                "status": "success",
                "message": message,
                "removed_count": removed_count,
                "removed_ids": removed_ids if removed_ids else None
            }
        else:
            return {
                "status": "warning" if "No preferences" in message else "error",
                "message": message,
                "removed_count": removed_count
            }
    
    except Exception as e:
        print(f"Error in clear_preferences endpoint: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete-preferences-by-name", tags=["users"])
async def delete_preferences_by_name(
    user_name: str
):
    """
    Delete user preferences from preferences.json file by user name
    
    This endpoint allows you to remove preferences by user name instead of ID.
    It will match partial names and delete all matching preferences.
    
    Parameters:
    - user_name: Name of the user whose preferences should be deleted
    """
    try:
        if not user_name:
            raise HTTPException(
                status_code=400,
                detail="User name must be provided"
            )
            
        preferences_file = os.path.join(engine.base_path, "preferences.json")
        if not os.path.exists(preferences_file):
            return {
                "status": "warning",
                "message": "No preferences file exists yet"
            }
            
        # Load existing preferences
        try:
            with open(preferences_file, 'r') as f:
                preferences = json.load(f)
                initial_count = len(preferences)
                print(f"Loaded {initial_count} preferences")
        except json.JSONDecodeError:
            return {
                "status": "error",
                "message": "Error decoding preferences.json file"
            }
        
        # Find and remove preferences with matching name
        original_length = len(preferences)
        new_preferences = []
        removed_ids = []
        
        for pref in preferences:
            name = pref.get('Name', '')
            if not name or user_name.lower() not in name.lower():
                new_preferences.append(pref)
            else:
                pref_id = pref.get('Unique_ID') or pref.get('unique_id')
                removed_ids.append(f"{name} ({pref_id})" if pref_id else name)
        
        # Save updated preferences
        with open(preferences_file, 'w') as f:
            json.dump(new_preferences, f, indent=4)
            
        if len(new_preferences) == original_length:
            return {
                "status": "warning",
                "message": f"No preferences found with name containing: {user_name}",
                "removed_count": 0
            }
        else:
            return {
                "status": "success",
                "message": f"Removed {original_length - len(new_preferences)} preferences with name containing: {user_name}",
                "removed_count": original_length - len(new_preferences),
                "removed_users": removed_ids
            }
    
    except Exception as e:
        print(f"Error deleting preferences by name: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/restore_base_model", tags=["model"])
async def restore_base_model(
    check_only: bool = Query(False, description="Only check if restore is possible without performing it"),
    keep_preferences: bool = Query(True, description="Keep preferences.json file after restore")
):
    """
    Restore the original base model (1000-sample model) and clean up learned data
    
    This endpoint allows you to:
    1. Remove the current learned model (nn_model.pkl)
    2. Remove model hashes (model_sample_hashes.json)
    3. Optionally remove preferences.json
    4. Restore the original base model
    
    Parameters:
    - check_only: Only check if restore is possible without performing it
    - keep_preferences: Keep preferences.json file after restore (default: True)
    """
    try:
        # Check file paths
        base_model_path = os.path.join(engine.base_path, "base_model.pkl")
        current_model_path = os.path.join(engine.base_path, "nn_model.pkl")
        model_hashes_path = os.path.join(engine.base_path, "model_sample_hashes.json")
        preferences_path = os.path.join(engine.base_path, "preferences.json")
        
        # Prepare status info
        status = {
            "base_model_exists": os.path.exists(base_model_path),
            "current_model_exists": os.path.exists(current_model_path),
            "model_hashes_exist": os.path.exists(model_hashes_path),
            "preferences_exist": os.path.exists(preferences_path),
            "can_restore": False,
            "files_to_remove": [],
            "files_to_keep": []
        }
        
        # Check if base model exists
        if not status["base_model_exists"]:
            return {
                "status": "error",
                "message": "Base model (base_model.pkl) not found. Cannot restore.",
                "details": status
            }
        
        # List files that will be affected
        if status["current_model_exists"]:
            status["files_to_remove"].append("nn_model.pkl")
        if status["model_hashes_exist"]:
            status["files_to_remove"].append("model_sample_hashes.json")
        if status["preferences_exist"] and not keep_preferences:
            status["files_to_remove"].append("preferences.json")
        elif status["preferences_exist"]:
            status["files_to_keep"].append("preferences.json")
            
        status["can_restore"] = True
        
        # If only checking, return status
        if check_only:
            return {
                "status": "success",
                "message": "Restore check completed",
                "details": status
            }
            
        # Perform restore
        try:
            # Remove current model files
            if os.path.exists(current_model_path):
                os.remove(current_model_path)
                print("Removed current model file")
                
            if os.path.exists(model_hashes_path):
                os.remove(model_hashes_path)
                print("Removed model hashes file")
                
            if not keep_preferences and os.path.exists(preferences_path):
                os.remove(preferences_path)
                print("Removed preferences file")
            
            # Load base model
            with open(base_model_path, 'rb') as f:
                base_model = pickle.load(f)
            
            # Update engine's model
            engine.knn_model = base_model
            
            # Save as current model
            with open(current_model_path, 'wb') as f:
                pickle.dump(base_model, f)
            
            print("Successfully restored base model")
            
            return {
                "status": "success",
                "message": "Successfully restored base model",
                "details": {
                    **status,
                    "files_removed": status["files_to_remove"],
                    "files_kept": status["files_to_keep"]
                }
            }
            
        except Exception as restore_error:
            print(f"Error during restore: {str(restore_error)}")
            return {
                "status": "error",
                "message": f"Error during restore: {str(restore_error)}",
                "details": status
            }
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Restore base model error: {str(e)}\n{error_details}")
        return {
            "status": "error",
            "message": f"Restore base model failed: {str(e)}",
            "error_details": error_details
        }

class ScoreWeights(BaseModel):
    knn_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for KNN similarity score (between 0 and 1). Sentiment weight will be (1 - knn_weight)"
    )

    class Config:
        schema_extra = {
            "example": {
                "knn_weight": 0.7
            }
        }

@app.post("/update_score_weights", tags=["model"])
async def update_score_weights(weights: ScoreWeights):
    """
    Update the weights used in final score calculation.
    
    The final score is calculated as:
    final_score = (knn_weight * similarity_score) + ((1 - knn_weight) * sentiment_score)
    
    Example input:
    {
        "knn_weight": 0.7
    }
    
    This means:
    - KNN similarity score will have 70% weight
    - Sentiment score will have 30% weight (1 - 0.7)
    """
    try:
        # Update the weights in the engine
        engine.score_weights = {
            'knn_weight': weights.knn_weight,
            'sentiment_weight': 1 - weights.knn_weight
        }
        
        # Save the weights to a file for persistence
        weights_file = os.path.join(engine.base_path, "score_weights.json")
        with open(weights_file, 'w') as f:
            json.dump(engine.score_weights, f, indent=2)
        
        return {
            "status": "success",
            "message": "Score weights updated successfully",
            "weights": {
                "knn_weight": weights.knn_weight,
                "sentiment_weight": 1 - weights.knn_weight
            }
        }
    except Exception as e:
        print(f"Error updating score weights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_score_weights", tags=["model"])
async def get_score_weights():
    """
    Get the current weights used in final score calculation.
    """
    try:
        return {
            "knn_weight": engine.score_weights.get('knn_weight', 0.7),
            "sentiment_weight": engine.score_weights.get('sentiment_weight', 0.3)
        }
    except Exception as e:
        print(f"Error getting score weights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 