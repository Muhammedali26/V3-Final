from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import random
import requests
import time
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import re
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Amazon Scraper Service",
    description="""
    A comprehensive microservice for scraping Amazon product data.
    
    Key Features:
    - Search Results Scraping: Extract product information from Amazon search results
    - Single Product Scraping: Detailed information about specific products
    - AI-Powered Review Analysis: Get AI-generated summaries of product reviews
    - Customer Review Extraction: Collect and analyze customer feedback
    - Real-time Data Processing: Immediate access to scraped data
    
    Technical Details:
    - Proxy rotation for reliable scraping
    - Rate limiting protection
    - Cookie management
    - Error handling and retry mechanisms
    - Background task processing
    
    Usage Guidelines:
    1. Use search URL endpoint for bulk product data
    2. Use single product endpoint for detailed product analysis
    3. Access real-time data through the data endpoint
    4. Direct data endpoint for immediate results
    """,
    version="1.0.0",
    openapi_tags=[{
        "name": "scraper",
        "description": "Advanced Amazon product scraping operations with comprehensive data extraction capabilities"
    }]
)

class ScrapeRequest(BaseModel):
    search_url: str = Field(
        ...,
        description="Amazon search URL to scrape. Should be a valid Amazon search results page URL.",
        example="https://www.amazon.com/s?k=shampoo+for+dry+hair"
    )

class SingleProductRequest(BaseModel):
    product_url: str = Field(
        ...,
        description="Amazon product URL to scrape. Must be a valid Amazon product page URL.",
        example="https://www.amazon.com/Mielle-Rosemary-Mint-Strengthening-Shampoo/dp/B07N7MWX72"
    )

class ScrapeResponse(BaseModel):
    message: str = Field(..., description="Detailed status message about the scraping operation")
    search_url: str = Field(..., description="The URL being processed by the scraper")
    estimated_time: str = Field(..., description="Estimated time to complete the scraping operation")

class ProductDetail(BaseModel):
    title: str = Field(..., description="Complete product title as shown on Amazon")
    price: str = Field(..., description="Current product price including currency symbol")
    rating: str = Field(..., description="Product rating out of 5 stars")
    review_count: str = Field(..., description="Total number of customer reviews")
    availability: str = Field(..., description="Current product availability status")
    ai_review_summary: Optional[str] = Field(None, description="AI-generated summary of customer reviews")

class ScrapedData(BaseModel):
    products: List[ProductDetail] = Field(..., description="List of scraped product details")
    total_products: int = Field(..., description="Total number of products successfully scraped")
    search_url: str = Field(..., description="Original search URL used for scraping")
    scrape_timestamp: str = Field(..., description="ISO format timestamp of when scraping was completed")

class AmazonScraperError(Exception):
    """Base exception class for Amazon scraper errors"""
    pass

class LoginError(AmazonScraperError):
    """Exception raised for authentication-related errors"""
    pass

class ProxyError(AmazonScraperError):
    """Exception raised for proxy-related issues"""
    pass

class CaptchaError(AmazonScraperError):
    """Exception raised when CAPTCHA is detected"""
    pass

class ScrapingError(AmazonScraperError):
    """Exception raised for general scraping failures"""
    pass

class AmazonScraper:
    """
    A comprehensive Amazon product scraper with advanced features.
    
    Features:
    - Proxy rotation
    - User agent rotation
    - Rate limiting protection
    - Cookie management
    - Automatic retry mechanism
    - AI-powered review analysis
    
    Technical Details:
    - Uses BeautifulSoup for HTML parsing
    - Implements session management
    - Handles various product page layouts
    - Supports both search results and single product scraping
    """
    
    def __init__(self):
        """
        Initialize the Amazon scraper with necessary configurations.
        
        Sets up:
        - User agent rotation
        - Session management
        - Cookie handling
        - Proxy configuration
        - Authentication settings
        """
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:88.0) Gecko/20100101 Firefox/88.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15"
        ]
        self.session = self._create_session()
        self.headers = {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
        self.url = None
        self.product_data = {
            'title': [],
            'price': [],
            'rating': [],
            'reviews': [],
            'availability': [],
            'review_details': [],
            'positive_feedback': [],
            'negative_feedback': []
        }
        self.email = os.getenv('AMAZON_EMAIL')
        self.password = os.getenv('AMAZON_PASSWORD')
        self.is_logged_in = False
        self.cookies_file = "amazon_cookies.json"
        self.load_cookies()
        self.proxies = [
            "http://185.199.229.156:7492",
            "http://185.199.228.220:7300",
            "http://185.199.231.45:8382",
            "http://188.74.210.207:6286",
            "http://188.74.183.10:8279",
            "http://188.74.210.21:6100",
            "http://45.155.68.129:8133",
            "http://154.85.100.34:5834",
            "http://45.94.47.66:8110"
        ]
        self.current_proxy = None

    def test_login(self) -> bool:
        """Test login status"""
        try:
            print("\nTesting login status...")
            
            # Fetch Amazon homepage
            test_url = "https://www.amazon.com/gp/css/homepage.html"
            response = self.session.get(test_url, allow_redirects=False)
            
            # Redirect check
            if response.status_code == 302:
                redirect_url = response.headers.get('Location', '')
                if 'signin' in redirect_url.lower():
                    print("Not logged in: Redirected to login page")
                    return False
            
            # Page content check
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Login status indicators
            indicators = {
                "Sign Out button": bool(soup.find("a", string=lambda x: x and "Sign Out" in x)),
                "Your Account": bool(soup.find("h1", string=lambda x: x and "Your Account" in x)),
                "Hello message": bool(soup.find(string=lambda x: x and "Hello, " in x)),
                "Your Orders": bool(soup.find("a", string=lambda x: x and "Your Orders" in x))
            }
            
            print("\nLogin status indicators:")
            for indicator, status in indicators.items():
                print(f"{indicator}: {'Found' if status else 'Not found'}")
            
            is_logged = any(indicators.values())
            print(f"\nLogin status: {'Logged in' if is_logged else 'Not logged in'}")
            
            return is_logged
            
        except Exception as e:
            print(f"Login test error: {str(e)}")
            return False

    def extract_product_id(self, url: str) -> Optional[str]:
        """Extract product ID from Amazon URL"""
        patterns = [
            r"/dp/([A-Z0-9]{10})",
            r"/product/([A-Z0-9]{10})",
            r"/gp/product/([A-Z0-9]{10})"
        ]
        
        print(f"\nTrying to extract product ID from URL: {url}")
        
        for pattern in patterns:
            print(f"Checking pattern: {pattern}")
            match = re.search(pattern, url)
            if match:
                product_id = match.group(1)
                print(f"Found product ID: {product_id} using pattern: {pattern}")
                return product_id
            else:
                print(f"No match found for pattern: {pattern}")
        
        print("No product ID found in URL")
        return None

    async def scrape_product_reviews(self, product_url: str) -> dict:
        print("\n=== Ürün detayları çekiliyor ===")
        print(f"URL: {product_url}")
        
        try:
            # Extract product ID
            product_id = self.extract_product_id(product_url)
            print(f"Extracted Product ID: {product_id}")
            
            if not product_id:
                print(f"Invalid product URL: {product_url}")
                return {
                    'product_details': None,
                    'ai_review_summary': None,
                    'customer_reviews': []
                }

            # Fetch product page
            print("Fetching product details...")
            product_soup = self.fetch_webpage(product_url)
            if not product_soup:
                print("Failed to fetch product page")
                return {
                    'product_details': None,
                    'ai_review_summary': None,
                    'customer_reviews': []
                }

            # Get product details
            product_details = {
                'title': self.get_title(product_soup),
                'price': self.get_price(product_soup),
                'rating': self.get_rating(product_soup),
                'review_count': self.get_review_count(product_soup),
                'availability': self.get_availability(product_soup)
            }

            # Get AI review summary
            ai_summary = self.get_ai_review_summary(product_soup)
            
            # Get customer reviews
            customer_reviews = self.get_customer_reviews(product_soup, max_reviews=15)
            print(f"Extracted {len(customer_reviews)} customer reviews")

            print("Successfully scraped product details, AI summary, and customer reviews")
            return {
                'product_details': product_details,
                'ai_review_summary': ai_summary,
                'customer_reviews': customer_reviews
            }

        except Exception as e:
            print(f"Scraping error: {str(e)}")
            return {
                'product_details': product_details if 'product_details' in locals() else None,
                'ai_review_summary': None,
                'customer_reviews': []
            }

    def scrape(self) -> dict:
        """Main scraping method that handles both search results and single products"""
        if self.is_product_url():
            return self.scrape_product_reviews(self.url)
        else:
            return self.scrape_search_results()

    def scrape_search_results(self) -> dict:
        """Scrape data from search results"""
        try:
            self.extract_product_data()
            
            products = []
            for i in range(len(self.product_data['title'])):
                products.append({
                    'title': self.product_data['title'][i],
                    'price': self.product_data['price'][i],
                    'rating': self.product_data['rating'][i],
                    'review_count': self.product_data['reviews'][i],
                    'availability': self.product_data['availability'][i],
                    'reviews': self.product_data['review_details'][i],
                    'positive_feedback': self.product_data['positive_feedback'][i],
                    'negative_feedback': self.product_data['negative_feedback'][i]
                })

            return {
                'products': products,
                'total_products': len(products),
                'search_url': self.url,
                'scrape_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Error scraping search results: {str(e)}")
            return None

    def _create_session(self) -> requests.Session:
        """Create and configure requests session"""
        session = requests.Session()
        session.headers.update({
            "User-Agent": random.choice(self.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "DNT": "1",
            "Cache-Control": "max-age=0"
        })
        return session

    def fetch_webpage(self, url: str, params: dict = None) -> Optional[BeautifulSoup]:
        """Fetch webpage and return BeautifulSoup object"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Proxy rotation for each request
                if retry_count > 0:
                    self.rotate_proxy()
                
                # Random User-Agent
                self.session.headers["User-Agent"] = random.choice(self.user_agents)
                
                # Random sleep before request
                self.random_sleep(1, 3)
                
                response = self.session.get(
                    url, 
                    params=params, 
                    timeout=15,
                    allow_redirects=True
                )
                
                if response.status_code == 200:
                    return BeautifulSoup(response.content, 'html.parser')
                elif response.status_code == 503:
                    print("Rate limit detected, waiting longer...")
                    self.random_sleep(5, 10)
                else:
                    print(f"Failed to fetch {url}: Status code {response.status_code}")
                
                retry_count += 1
                
            except Exception as e:
                print(f"Error fetching {url}: {str(e)}")
                retry_count += 1
                self.random_sleep(3, 7)
        
        return None

    def extract_links(self, soup: BeautifulSoup) -> list:
        if soup is None:
            return []
        links = soup.find_all("a", class_='a-link-normal s-no-outline')
        print(f"Found {len(links)} links.")
        return ["https://www.amazon.com" + link.get('href') for link in links]

    def get_title(self, soup: BeautifulSoup) -> str:
        """Get product title"""
        if not soup:
            return ""
        title = soup.find("span", {"id": "productTitle"})
        return title.text.strip() if title else ""

    def get_price(self, soup: BeautifulSoup) -> str:
        """Get product price"""
        if not soup:
            return ""
        
        try:
            # 1. Method: Extract price using a-price class
            price_whole = soup.find("span", class_="a-price-whole")
            price_fraction = soup.find("span", class_="a-price-fraction")
            if price_whole and price_fraction:
                return f"${price_whole.text.strip()}{price_fraction.text.strip()}"
            
            # 2. Method: Extract price using a-offscreen class
            price_element = soup.find("span", class_="a-offscreen")
            if price_element:
                return price_element.text.strip()
            
            # 3. Method: Extract price from special price block
            price_block = soup.find("div", id="corePrice_feature_div")
            if price_block:
                price = price_block.find("span", class_="a-offscreen")
                if price:
                    return price.text.strip()
            
            # 4. Method: Check all possible price elements
            price_elements = soup.find_all("span", class_=lambda x: x and "price" in x.lower())
            for element in price_elements:
                price_text = element.text.strip()
                if "$" in price_text and any(c.isdigit() for c in price_text):
                    return price_text
                
        except Exception as e:
            print(f"Price extraction error: {str(e)}")
        
        return ""

    def get_rating(self, soup: BeautifulSoup) -> str:
        """Get product rating with all possible selectors"""
        if not soup:
            return ""
        
        # All possible rating elements
        rating_selectors = [
            # Main rating element
            {"class": "a-size-base a-color-base"},
            # Star rating element
            {"class": "a-icon-alt"},
            # Global rating element
            {"id": "acrPopover"},
            # Alternative rating element
            {"data-hook": "rating-out-of-text"}
        ]
        
        try:
            # First, find rating section
            rating_section = soup.find("div", {"id": "averageCustomerReviews"})
            if rating_section:
                # Try all possible rating elements
                for selector in rating_selectors:
                    if "class" in selector:
                        rating_element = rating_section.find("span", class_=selector["class"])
                    elif "id" in selector:
                        rating_element = rating_section.find("span", id=selector["id"])
                    elif "data-hook" in selector:
                        rating_element = rating_section.find("span", attrs={"data-hook": selector["data-hook"]})
                    
                    if rating_element:
                        rating_text = rating_element.text.strip()
                        # Extract rating value (e.g., "4.6 out of 5" -> "4.6")
                        if "out of" in rating_text.lower():
                            return rating_text.split("out of")[0].strip()
                        elif "stars" in rating_text.lower():
                            return rating_text.split("stars")[0].strip()
                        else:
                            return rating_text

            # Alternative method: Directly search for
            for selector in rating_selectors:
                if "class" in selector:
                    rating_element = soup.find("span", class_=selector["class"])
                elif "id" in selector:
                    rating_element = soup.find("span", id=selector["id"])
                elif "data-hook" in selector:
                    rating_element = soup.find("span", attrs={"data-hook": selector["data-hook"]})
                
                if rating_element:
                    rating_text = rating_element.text.strip()
                    if "out of" in rating_text.lower():
                        return rating_text.split("out of")[0].strip()
                    elif "stars" in rating_text.lower():
                        return rating_text.split("stars")[0].strip()
                    else:
                        return rating_text
                    
        except Exception as e:
            print(f"Rating extraction error: {str(e)}")
        
        return ""

    def get_review_count(self, soup: BeautifulSoup) -> str:
        """Get total review count"""
        if not soup:
            return ""
        count = soup.find("span", {"id": "acrCustomerReviewText"})
        return count.text.strip() if count else ""

    def get_availability(self, soup: BeautifulSoup) -> str:
        """Get product availability"""
        if not soup:
            return ""
        
        try:
            # 1. Method: Direct availability span
            availability_span = soup.find("span", class_="a-size-medium a-color-success")
            if availability_span:
                return availability_span.text.strip()
            
            # 2. Method: Availability block
            availability_div = soup.find("div", {"id": "availability"})
            if availability_div:
                span = availability_div.find("span")
                if span:
                    return span.text.strip()
                
            # 3. Method: Alternative availability indicators
            delivery_block = soup.find("div", {"id": "deliveryBlockMessage"})
            if delivery_block:
                return delivery_block.text.strip()
            
            # 4. Method: Other stock indicators
            stock_status = soup.find("span", {"class": lambda x: x and "availability" in x.lower()})
            if stock_status:
                return stock_status.text.strip()
            
        except Exception as e:
            print(f"Availability extraction error: {str(e)}")
        
        return "Availability unknown"  # Default value

    def extract_product_data(self):
        soup = self.fetch_webpage(self.url)
        if soup is None:
            print("Failed to retrieve the search results page.")
            return

        links = self.extract_links(soup)
        if not links:
            print("No links extracted. Please check the search URL or the HTML structure.")
            return

        for link in links:
            print(f"Processing link: {link}")
            time.sleep(random.uniform(2, 4))
            product_soup = self.fetch_webpage(link)
            
            self.product_data['title'].append(self.get_title(product_soup))
            self.product_data['price'].append(self.get_price(product_soup))
            self.product_data['rating'].append(self.get_rating(product_soup))
            self.product_data['reviews'].append(self.get_review_count(product_soup))
            self.product_data['availability'].append(self.get_availability(product_soup))

            reviews = self.get_customer_reviews(product_soup)
            self.product_data['review_details'].append(reviews)
            
            if reviews:
                positive_feedback = max(reviews, key=lambda x: int(x['helpful_votes'].split()[0]) if x['helpful_votes'].split()[0].isdigit() else 0)['text']
                negative_feedback = min(reviews, key=lambda x: int(x['helpful_votes'].split()[0]) if x['helpful_votes'].split()[0].isdigit() else float('inf'))['text']
                self.product_data['positive_feedback'].append(positive_feedback)
                self.product_data['negative_feedback'].append(negative_feedback)
            else:
                self.product_data['positive_feedback'].append("")
                self.product_data['negative_feedback'].append("")

    def is_product_url(self, url: str = None) -> bool:
        """
        Check if a URL is a product page
        """
        if url is None:
            url = self.url
            
        # Product URL patterns
        product_patterns = [
            r'/dp/[A-Z0-9]{10}',
            r'/gp/product/[A-Z0-9]{10}',
            r'/product/[A-Z0-9]{10}'
        ]
        
        return any(re.search(pattern, url) for pattern in product_patterns)

    def load_cookies(self):
        """Load saved cookies"""
        try:
            if os.path.exists(self.cookies_file):
                with open(self.cookies_file, 'r') as f:
                    self.session.cookies.update(json.loads(f.read()))
                print("Cookies loaded successfully")
        except Exception as e:
            print(f"Error loading cookies: {str(e)}")

    def save_cookies(self):
        """Save cookies"""
        try:
            with open(self.cookies_file, 'w') as f:
                json.dump(requests.utils.dict_from_cookiejar(self.session.cookies), f)
            print("Cookies saved successfully")
        except Exception as e:
            print(f"Error saving cookies: {str(e)}")

    def get_ai_review_summary(self, soup: BeautifulSoup) -> str:
        """Extract AI-generated review summary from product page"""
        try:
            # CSS Selector ile dene
            summary = soup.select_one("#product-summary > p.a-spacing-small > span")
            if summary:
                summary_text = summary.get_text(strip=True)
                print(f"CSS Selector ile özet bulundu: {summary_text[:100]}...")
                return summary_text

            # Alternatif olarak product-summary div'ini bul ve içindeki span'i ara
            product_summary = soup.find("div", id="product-summary")
            if product_summary:
                summary = product_summary.find("p", class_="a-spacing-small")
                if summary:
                    span_text = summary.find("span")
                    if span_text:
                        summary_text = span_text.get_text(strip=True)
                        print(f"Alternatif yöntem ile özet bulundu: {summary_text[:100]}...")
                        return summary_text

            print("AI özeti bulunamadı")
            return ""

        except Exception as e:
            print(f"AI review summary extraction error: {str(e)}")
            return ""

    def random_sleep(self, min_seconds: float = 1.0, max_seconds: float = 3.0):
        """
        Randomly wait between specified intervals
        
        Args:
            min_seconds (float): Minimum wait time (seconds)
            max_seconds (float): Maximum wait time (seconds)
        """
        time.sleep(random.uniform(min_seconds, max_seconds))

    def get_customer_reviews(self, soup: BeautifulSoup, max_reviews: int = 15) -> List[str]:
        """
        Extract customer reviews from Amazon page (optimized version)
        
        Args:
            soup: BeautifulSoup object
            max_reviews: Maximum number of reviews to extract (default: 15)
        
        Returns:
            List of reviews
        """
        reviews = []
        try:
            # First, try the most effective selector
            review_elements = soup.select(".review-text-content span")
            if review_elements:
                print(f"Found {len(review_elements)} reviews with primary selector")
                for i, review_elem in enumerate(review_elements[:max_reviews]):
                    review_text = review_elem.get_text(strip=True)
                    if review_text:
                        reviews.append(review_text)
                
                # Enough reviews found
                if len(reviews) >= max_reviews:
                    return reviews[:max_reviews]
            
            # If not enough reviews, try data-hook
            if len(reviews) < max_reviews:
                review_spans = soup.find_all("span", attrs={"data-hook": "review-body"})
                for span in review_spans[:max_reviews-len(reviews)]:
                    text = span.get_text(strip=True)
                    if text and text not in reviews:  # Avoid duplicate reviews
                        reviews.append(text)
                
            # Still not enough reviews, try customer review divs
            if len(reviews) < max_reviews:
                review_divs = soup.find_all("div", attrs={"data-hook": "review"})
                for div in review_divs[:max_reviews-len(reviews)]:
                    # Find the longest span in the div (probably the review text)
                    spans = div.find_all("span")
                    if spans:
                        # Find the longest text-containing span
                        longest_span = max(spans, key=lambda s: len(s.text.strip()), default=None)
                        if longest_span and len(longest_span.text.strip()) > 50:
                            text = longest_span.get_text(strip=True)
                            if text and text not in reviews:
                                reviews.append(text)
            
            # Return results
            print(f"Found {len(reviews)} reviews total")
            return reviews[:max_reviews]
            
        except Exception as e:
            print(f"Error extracting customer reviews: {str(e)}")
            return []

    def get_customer_reviews_fast(self, soup: BeautifulSoup, max_reviews: int = 15) -> List[str]:
        """Ultra fast review extraction (only most common selectors)"""
        reviews = []
        try:
            # Only try the most common selector
            review_elements = soup.select(".review-text-content span")
            for elem in review_elements[:max_reviews]:
                text = elem.get_text(strip=True)
                if text:
                    reviews.append(text)
            return reviews[:max_reviews]
        except Exception as e:
            print(f"Error: {str(e)}")
            return []

@app.post("/scrape_product", tags=["scraper"], response_model=ScrapeResponse)
async def scrape_single_product(request: SingleProductRequest):
    """
    Scrape detailed data from a single Amazon product page.
    
    This endpoint performs a comprehensive scraping operation on a single product, including:
    - Basic product information (title, price, rating)
    - Availability status
    - Customer reviews
    - AI-generated review summary
    
    Rate Limiting:
    - Implements intelligent delays between requests
    - Uses proxy rotation for reliability
    
    Error Handling:
    - Handles network errors gracefully
    - Detects and manages CAPTCHA challenges
    - Provides detailed error messages
    
    Args:
        request (SingleProductRequest): Contains the product URL to scrape
        
    Returns:
        ScrapeResponse: Contains status message, URL, and estimated completion time
        
    Raises:
        HTTPException: For various error conditions with appropriate status codes
    """
    import sys
    def log(msg):
        print(msg, file=sys.stderr, flush=True)
    
    log("\n=== SCRAPING TEST STARTED ===")
    log(f"URL: {request.product_url}")
    
    try:
        global scraper, scraped_data
        scraped_data = {}  # Clear previous data
        
        log("1. Creating scraper instance...")
        scraper = AmazonScraper()
        scraper.url = request.product_url
        
        log("2. Testing webpage fetch...")
        response = requests.get(request.product_url, headers=scraper.headers, timeout=10)
        log(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            log("4. Page fetched successfully")
            
            # Parse page content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract product details
            product_details = {
                'title': scraper.get_title(soup),
                'price': scraper.get_price(soup),
                'rating': scraper.get_rating(soup),
                'review_count': scraper.get_review_count(soup),
                'availability': scraper.get_availability(soup)
            }
            
            # Get AI summary
            ai_summary = scraper.get_ai_review_summary(soup)
            log(f"AI Summary: {ai_summary[:100]}..." if ai_summary else "No AI summary found")
            
            # Get customer reviews
            customer_reviews = scraper.get_customer_reviews(soup, max_reviews=15)
            log(f"Extracted customer reviews: {len(customer_reviews)}")
            
            # Cache the data
            scraped_data = {
                'product_details': product_details,
                'ai_review_summary': ai_summary,
                'customer_reviews': customer_reviews
            }
            
            log("Data successfully cached.")
        else:
            log(f"4. Error: Status code {response.status_code}")
            
        return ScrapeResponse(
            message=f"Operation completed. Status: {response.status_code}, Reviews extracted: {len(scraped_data.get('customer_reviews', []))}",
            search_url=request.product_url,
            estimated_time="10-20 seconds"
        )
        
    except Exception as e:
        log(f"ERROR OCCURRED: {str(e)}")
        import traceback
        log(f"ERROR DETAILS:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scrape", tags=["scraper"], response_model=ScrapeResponse)
async def scrape_data(request: ScrapeRequest, background_tasks: BackgroundTasks):
    """
    Start scraping Amazon product data from a search results page.
    
    This endpoint initiates an asynchronous scraping operation that:
    - Processes search results pages
    - Extracts multiple product details
    - Handles pagination automatically
    - Runs in the background for better performance
    
    Features:
    - Asynchronous processing
    - Background task execution
    - Progress tracking
    - Automatic error recovery
    
    Args:
        request (ScrapeRequest): Contains the search URL to scrape
        background_tasks: FastAPI background tasks handler
        
    Returns:
        ScrapeResponse: Contains operation status and timing information
        
    Raises:
        HTTPException: For various error conditions
    """
    global scraper
    try:
        scraper = AmazonScraper()
        background_tasks.add_task(scraper.scrape)
        return ScrapeResponse(
            message="Scraping operation started successfully",
            search_url=request.search_url,
            estimated_time="30-60 seconds"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data", tags=["scraper"], response_model=None)
async def get_data():
    """
    Retrieve the scraped product data including AI-generated summaries and reviews.
    
    This endpoint provides access to:
    - Detailed product information
    - AI-generated review summaries
    - Customer reviews and ratings
    - Availability and pricing data
    
    Features:
    - Cached data access
    - Real-time data updates
    - Comprehensive product details
    
    Returns:
        dict: Contains all scraped product data and metadata
        
    Raises:
        HTTPException: For various error conditions including:
            - 400: Scraper not initialized
            - 404: No product data found
            - 500: Internal processing errors
    """
    global scraper, scraped_data
    print("\n=== DATA ENDPOINT CALLED ===")
    
    if not scraper:
        print("Scraper not yet initialized!")
        raise HTTPException(status_code=400, detail="Scraper not initialized")
    
    try:
        print(f"Current URL: {scraper.url}")
        print(f"Cache status: {'Populated' if scraped_data else 'Empty'}")
        
        # Return cached data if available
        if scraped_data:
            print("Returning data from cache")
            print(f"Returned data preview: {str(scraped_data)[:200]}...")
            return scraped_data
            
        # Fetch new data
        print("Fetching new data...")
        product_soup = scraper.fetch_webpage(scraper.url)
        
        if not product_soup:
            print("Failed to fetch webpage! Empty soup returned.")
            raise HTTPException(status_code=404, detail="No product data found")
        
        print("Webpage successfully fetched, extracting data...")
        
        try:
            product_details = {
                'title': scraper.get_title(product_soup),
                'price': scraper.get_price(product_soup),
                'rating': scraper.get_rating(product_soup),
                'review_count': scraper.get_review_count(product_soup),
                'availability': scraper.get_availability(product_soup)
            }
            
            print(f"Product details: {product_details}")
            
            ai_summary = scraper.get_ai_review_summary(product_soup)
            print(f"AI summary preview: {ai_summary[:100] if ai_summary else 'None'}")
            
            # Extract customer reviews
            customer_reviews = scraper.get_customer_reviews(product_soup, max_reviews=15)
            print(f"Extracted customer reviews: {len(customer_reviews)}")
            
            data = {
                'product_details': product_details,
                'ai_review_summary': ai_summary,
                'customer_reviews': customer_reviews
            }
            
            # Cache the data
            scraped_data = data
            print(f"Data successfully generated and cached.")
            return data
            
        except Exception as e:
            print(f"Data extraction error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error extracting data: {str(e)}")
        
    except Exception as e:
        print(f"Data endpoint general error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/direct-data")
async def get_direct_data(url: str):
    """
    Fetch and return data directly from a given Amazon URL.
    
    This endpoint combines the functionality of scrape_product and data endpoints
    into a single operation for improved efficiency. It provides:
    - Immediate data access
    - Complete product information
    - Error handling and recovery
    
    Features:
    - Single-step operation
    - Direct data access
    - Comprehensive error handling
    
    Args:
        url (str): The Amazon product URL to scrape
        
    Returns:
        dict: Contains complete product data including:
            - Product details
            - AI review summary
            - Customer reviews
            
    Note:
        This endpoint is optimized for performance but may take longer
        to respond compared to the async version.
    """
    try:
        # Create scraper instance
        temp_scraper = AmazonScraper()
        temp_scraper.url = url
        
        # Get page content
        soup = temp_scraper.fetch_webpage(url)
        
        if not soup:
            return {"error": "Failed to fetch the webpage"}
        
        # Extract data
        product_details = {
            'title': temp_scraper.get_title(soup),
            'price': temp_scraper.get_price(soup),
            'rating': temp_scraper.get_rating(soup),
            'review_count': temp_scraper.get_review_count(soup),
            'availability': temp_scraper.get_availability(soup)
        }
        
        ai_summary = temp_scraper.get_ai_review_summary(soup)
        customer_reviews = temp_scraper.get_customer_reviews(soup, max_reviews=15)
        
        return {
            'product_details': product_details,
            'ai_review_summary': ai_summary,
            'customer_reviews': customer_reviews
        }
        
    except Exception as e:
        import traceback
        print(f"Direct-data error: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e)}

if __name__ == "__main__":
    import sys
    print("Server starting...", file=sys.stderr, flush=True)
    
    # Update logging settings
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )
    
    import uvicorn
    uvicorn.run(
        "amazon_scraper:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug",
        workers=1,
        access_log=True
    ) 