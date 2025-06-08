# sentiment_analysis.py - Advanced Sentiment Analysis Module
import logging
import asyncio
import json
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import aiohttp
import random

# Optional sentiment analysis libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class SentimentSignal:
    """Individual sentiment signal"""
    source: str
    sentiment_score: float  # -100 to +100
    confidence: float  # 0-1
    volume: int  # Number of mentions/articles
    keywords: List[str]
    timestamp: datetime
    trend_direction: str  # UP, DOWN, NEUTRAL
    emotional_intensity: str  # LOW, MEDIUM, HIGH
    contrarian_indicator: bool = False

@dataclass
class SentimentAnalysis:
    """Complete sentiment analysis result"""
    overall_sentiment: float  # -100 to +100
    confidence: float
    trend_momentum: float
    signals: List[SentimentSignal]
    key_themes: List[str]
    retail_sentiment: str
    institutional_sentiment: str
    social_buzz_level: str
    fear_greed_indicator: str
    contrarian_signals: List[str]

class AdvancedSentimentAnalyzer:
    """Advanced Multi-Source Sentiment Analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sentiment_cache = {}
        self.cache_duration = 300  # 5 minutes
        self.trend_history = {}
        
        # Initialize sentiment analyzers
        self.vader_analyzer = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None
        
        # News and social media simulators
        self.news_simulator = NewsSimulator()
        self.social_simulator = SocialMediaSimulator()
        self.reddit_simulator = RedditSimulator()
        self.twitter_simulator = TwitterSimulator()
        
        # Sentiment weights by source
        self.source_weights = {
            'news': config.get('sentiment_weights', {}).get('news', 0.4),
            'reddit': config.get('sentiment_weights', {}).get('reddit', 0.35),
            'twitter': config.get('sentiment_weights', {}).get('twitter', 0.25),
            'social_general': 0.15,
            'institutional': 0.1
        }
        
        logger.info("Advanced Sentiment Analyzer initialized")
        logger.info(f"TextBlob: {'Available' if TEXTBLOB_AVAILABLE else 'Not Available'}")
        logger.info(f"VADER: {'Available' if VADER_AVAILABLE else 'Not Available'}")
    
    async def analyze(self, symbol: str, quote_data: Dict[str, Any], 
                     trends_data: Optional[Dict] = None) -> SentimentAnalysis:
        """Complete sentiment analysis from all sources"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{time.time() // self.cache_duration}"
            if cache_key in self.sentiment_cache:
                logger.debug(f"Using cached sentiment for {symbol}")
                return self.sentiment_cache[cache_key]
            
            logger.debug(f"Analyzing sentiment for {symbol}")
            
            # Collect sentiment signals from all sources
            signals = []
            
            # News sentiment
            news_signal = await self._analyze_news_sentiment(symbol, quote_data)
            if news_signal:
                signals.append(news_signal)
            
            # Social media sentiment
            social_signals = await self._analyze_social_sentiment(symbol, quote_data, trends_data)
            signals.extend(social_signals)
            
            # Reddit sentiment
            reddit_signal = await self._analyze_reddit_sentiment(symbol, quote_data)
            if reddit_signal:
                signals.append(reddit_signal)
            
            # Twitter sentiment
            twitter_signal = await self._analyze_twitter_sentiment(symbol, quote_data)
            if twitter_signal:
                signals.append(twitter_signal)
            
            # Institutional sentiment estimation
            institutional_signal = await self._analyze_institutional_sentiment(symbol, quote_data)
            if institutional_signal:
                signals.append(institutional_signal)
            
            # Calculate overall sentiment
            overall_sentiment, confidence, trend_momentum = self._calculate_overall_sentiment(signals)
            
            # Extract key themes and insights
            key_themes = self._extract_key_themes(signals)
            retail_sentiment = self._determine_retail_sentiment(signals)
            institutional_sentiment = self._determine_institutional_sentiment(signals)
            social_buzz_level = self._calculate_social_buzz_level(signals)
            fear_greed_indicator = self._calculate_fear_greed_indicator(overall_sentiment, signals)
            contrarian_signals = self._identify_contrarian_signals(signals, overall_sentiment)
            
            # Update trend history
            self._update_trend_history(symbol, overall_sentiment)
            
            result = SentimentAnalysis(
                overall_sentiment=overall_sentiment,
                confidence=confidence,
                trend_momentum=trend_momentum,
                signals=signals,
                key_themes=key_themes,
                retail_sentiment=retail_sentiment,
                institutional_sentiment=institutional_sentiment,
                social_buzz_level=social_buzz_level,
                fear_greed_indicator=fear_greed_indicator,
                contrarian_signals=contrarian_signals
            )
            
            # Cache result
            self.sentiment_cache[cache_key] = result
            
            logger.debug(f"Sentiment analysis for {symbol}: {overall_sentiment:.1f} ({confidence:.1%} confidence)")
            
            return result
            
        except Exception as e:
            logger.error(f"Sentiment analysis error for {symbol}: {e}")
            return self._generate_fallback_sentiment(symbol)
    
    async def _analyze_news_sentiment(self, symbol: str, quote_data: Dict[str, Any]) -> Optional[SentimentSignal]:
        """Analyze news sentiment"""
        try:
            # Get simulated news data
            news_data = await self.news_simulator.get_news_sentiment(symbol, quote_data)
            
            if not news_data:
                return None
            
            # Analyze news headlines and content
            sentiment_scores = []
            keywords = []
            
            for article in news_data.get('articles', []):
                headline = article.get('headline', '')
                content = article.get('summary', '')
                
                # Extract keywords
                article_keywords = self._extract_keywords(headline + " " + content, symbol)
                keywords.extend(article_keywords)
                
                # Calculate sentiment
                if headline or content:
                    text_sentiment = self._analyze_text_sentiment(headline + " " + content)
                    
                    # Weight by article relevance and recency
                    relevance = article.get('relevance_score', 1.0)
                    hours_old = article.get('hours_old', 1)
                    time_weight = max(0.1, 1.0 - (hours_old / 24))  # Decay over 24 hours
                    
                    weighted_sentiment = text_sentiment * relevance * time_weight
                    sentiment_scores.append(weighted_sentiment)
            
            if not sentiment_scores:
                return None
            
            # Calculate aggregated news sentiment
            avg_sentiment = np.mean(sentiment_scores)
            confidence = min(1.0, len(sentiment_scores) / 10)  # Higher confidence with more articles
            
            # Determine trend direction
            trend_direction = "UP" if avg_sentiment > 0.1 else "DOWN" if avg_sentiment < -0.1 else "NEUTRAL"
            
            # Emotional intensity based on sentiment magnitude and volume
            intensity_score = abs(avg_sentiment) * len(sentiment_scores)
            emotional_intensity = "HIGH" if intensity_score > 5 else "MEDIUM" if intensity_score > 2 else "LOW"
            
            return SentimentSignal(
                source="news",
                sentiment_score=avg_sentiment * 100,  # Convert to -100 to +100 scale
                confidence=confidence,
                volume=len(news_data.get('articles', [])),
                keywords=list(set(keywords))[:10],  # Top 10 unique keywords
                timestamp=datetime.now(),
                trend_direction=trend_direction,
                emotional_intensity=emotional_intensity,
                contrarian_indicator=self._detect_contrarian_news(news_data, avg_sentiment)
            )
            
        except Exception as e:
            logger.error(f"News sentiment analysis error: {e}")
            return None
    
    async def _analyze_social_sentiment(self, symbol: str, quote_data: Dict[str, Any], 
                                       trends_data: Optional[Dict] = None) -> List[SentimentSignal]:
        """Analyze general social media sentiment"""
        signals = []
        
        try:
            # Get Google Trends-based sentiment
            if trends_data:
                trends_signal = await self._analyze_trends_sentiment(symbol, trends_data)
                if trends_signal:
                    signals.append(trends_signal)
            
            # Get general social media sentiment
            social_data = await self.social_simulator.get_social_sentiment(symbol, quote_data)
            if social_data:
                social_signal = self._process_social_data(symbol, social_data, "social_general")
                if social_signal:
                    signals.append(social_signal)
            
        except Exception as e:
            logger.error(f"Social sentiment analysis error: {e}")
        
        return signals
    
    async def _analyze_reddit_sentiment(self, symbol: str, quote_data: Dict[str, Any]) -> Optional[SentimentSignal]:
        """Analyze Reddit sentiment"""
        try:
            reddit_data = await self.reddit_simulator.get_reddit_sentiment(symbol, quote_data)
            
            if not reddit_data:
                return None
            
            return self._process_social_data(symbol, reddit_data, "reddit")
            
        except Exception as e:
            logger.error(f"Reddit sentiment analysis error: {e}")
            return None
    
    async def _analyze_twitter_sentiment(self, symbol: str, quote_data: Dict[str, Any]) -> Optional[SentimentSignal]:
        """Analyze Twitter sentiment"""
        try:
            twitter_data = await self.twitter_simulator.get_twitter_sentiment(symbol, quote_data)
            
            if not twitter_data:
                return None
            
            return self._process_social_data(symbol, twitter_data, "twitter")
            
        except Exception as e:
            logger.error(f"Twitter sentiment analysis error: {e}")
            return None
    
    async def _analyze_institutional_sentiment(self, symbol: str, quote_data: Dict[str, Any]) -> Optional[SentimentSignal]:
        """Analyze institutional sentiment (estimated)"""
        try:
            current_price = float(quote_data.get('price', 100))
            change_percent = float(quote_data.get('change_percent', 0))
            volume = quote_data.get('volume', 0)
            
            # Estimate institutional sentiment based on price action and volume
            # Large volume + small price change = institutional accumulation/distribution
            # Large price change + high volume = institutional momentum
            
            avg_volume = 25000000  # Estimated average volume
            volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
            
            # Calculate institutional sentiment score
            institutional_score = 0
            
            # High volume, small change = accumulation (bullish) or distribution (bearish)
            if volume_ratio > 1.5 and abs(change_percent) < 1:
                if change_percent > 0:
                    institutional_score = 30  # Quiet accumulation
                else:
                    institutional_score = -30  # Quiet distribution
            
            # High volume, large change = momentum
            elif volume_ratio > 1.5 and abs(change_percent) > 2:
                institutional_score = change_percent * 10  # Amplify with volume
            
            # Low volume = neutral institutional sentiment
            elif volume_ratio < 0.8:
                institutional_score = 0
            
            # Normal volume = follow price action but muted
            else:
                institutional_score = change_percent * 5
            
            # Bound the score
            institutional_score = max(-50, min(50, institutional_score))
            
            confidence = min(1.0, volume_ratio / 2)  # Higher volume = higher confidence
            trend_direction = "UP" if institutional_score > 5 else "DOWN" if institutional_score < -5 else "NEUTRAL"
            
            return SentimentSignal(
                source="institutional",
                sentiment_score=institutional_score,
                confidence=confidence,
                volume=int(volume),
                keywords=["institutional_flow", "volume_analysis"],
                timestamp=datetime.now(),
                trend_direction=trend_direction,
                emotional_intensity="LOW",  # Institutions are less emotional
                contrarian_indicator=False
            )
            
        except Exception as e:
            logger.error(f"Institutional sentiment analysis error: {e}")
            return None
    
    async def _analyze_trends_sentiment(self, symbol: str, trends_data: Dict) -> Optional[SentimentSignal]:
        """Analyze Google Trends-based sentiment"""
        try:
            keyword_data = trends_data.get('keyword_data', {})
            
            if not keyword_data:
                return None
            
            # Aggregate momentum from all keywords
            total_momentum = 0
            total_weight = 0
            keywords = []
            
            for keyword, data in keyword_data.items():
                momentum = data.get('momentum', 0)
                current_value = data.get('current_value', 50)
                
                # Weight by search volume (current_value)
                weight = current_value / 100
                total_momentum += momentum * weight
                total_weight += weight
                
                keywords.append(keyword)
            
            if total_weight == 0:
                return None
            
            avg_momentum = total_momentum / total_weight
            
            # Convert momentum to sentiment score
            sentiment_score = max(-50, min(50, avg_momentum))
            
            confidence = min(1.0, total_weight)
            trend_direction = "UP" if sentiment_score > 5 else "DOWN" if sentiment_score < -5 else "NEUTRAL"
            
            # Emotional intensity based on momentum magnitude
            emotional_intensity = "HIGH" if abs(avg_momentum) > 30 else "MEDIUM" if abs(avg_momentum) > 15 else "LOW"
            
            return SentimentSignal(
                source="google_trends",
                sentiment_score=sentiment_score,
                confidence=confidence,
                volume=len(keyword_data),
                keywords=keywords,
                timestamp=datetime.now(),
                trend_direction=trend_direction,
                emotional_intensity=emotional_intensity,
                contrarian_indicator=abs(avg_momentum) > 40  # Extreme trends can be contrarian
            )
            
        except Exception as e:
            logger.error(f"Trends sentiment analysis error: {e}")
            return None
    
    def _process_social_data(self, symbol: str, social_data: Dict, source: str) -> Optional[SentimentSignal]:
        """Process social media data into sentiment signal"""
        try:
            posts = social_data.get('posts', [])
            
            if not posts:
                return None
            
            sentiment_scores = []
            keywords = []
            total_engagement = 0
            
            for post in posts:
                text = post.get('text', '')
                engagement = post.get('engagement', 1)
                
                if text:
                    # Extract keywords
                    post_keywords = self._extract_keywords(text, symbol)
                    keywords.extend(post_keywords)
                    
                    # Calculate sentiment
                    sentiment = self._analyze_text_sentiment(text)
                    
                    # Weight by engagement
                    weighted_sentiment = sentiment * (1 + np.log(engagement + 1))
                    sentiment_scores.append(weighted_sentiment)
                    total_engagement += engagement
            
            if not sentiment_scores:
                return None
            
            # Calculate aggregated sentiment
            avg_sentiment = np.mean(sentiment_scores)
            
            # Confidence based on number of posts and total engagement
            confidence = min(1.0, (len(sentiment_scores) * np.log(total_engagement + 1)) / 100)
            
            trend_direction = "UP" if avg_sentiment > 0.1 else "DOWN" if avg_sentiment < -0.1 else "NEUTRAL"
            
            # Emotional intensity for social media tends to be higher
            intensity_score = abs(avg_sentiment) * len(sentiment_scores)
            emotional_intensity = "HIGH" if intensity_score > 3 else "MEDIUM" if intensity_score > 1 else "LOW"
            
            return SentimentSignal(
                source=source,
                sentiment_score=avg_sentiment * 100,
                confidence=confidence,
                volume=len(posts),
                keywords=list(set(keywords))[:10],
                timestamp=datetime.now(),
                trend_direction=trend_direction,
                emotional_intensity=emotional_intensity,
                contrarian_indicator=self._detect_contrarian_social(social_data, avg_sentiment)
            )
            
        except Exception as e:
            logger.error(f"Social data processing error: {e}")
            return None
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using available libraries"""
        if not text:
            return 0.0
        
        sentiments = []
        
        # VADER Sentiment (if available)
        if self.vader_analyzer:
            vader_score = self.vader_analyzer.polarity_scores(text)
            compound_score = vader_score['compound']
            sentiments.append(compound_score)
        
        # TextBlob Sentiment (if available)
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                textblob_score = blob.sentiment.polarity
                sentiments.append(textblob_score)
            except:
                pass
        
        # Rule-based sentiment (fallback)
        rule_based_score = self._rule_based_sentiment(text)
        sentiments.append(rule_based_score)
        
        # Return average of available sentiment scores
        return np.mean(sentiments) if sentiments else 0.0
    
    def _rule_based_sentiment(self, text: str) -> float:
        """Simple rule-based sentiment analysis"""
        text = text.lower()
        
        # Positive words
        positive_words = [
            'bull', 'bullish', 'buy', 'moon', 'rocket', 'pump', 'green', 'up', 'rise', 'surge',
            'breakout', 'rally', 'strong', 'good', 'great', 'excellent', 'positive', 'gain',
            'profit', 'win', 'success', 'beat', 'exceed', 'optimistic', 'confident'
        ]
        
        # Negative words
        negative_words = [
            'bear', 'bearish', 'sell', 'dump', 'crash', 'red', 'down', 'fall', 'drop', 'plunge',
            'breakdown', 'weak', 'bad', 'terrible', 'negative', 'loss', 'lose', 'fail',
            'miss', 'disappoint', 'pessimistic', 'worried', 'concern', 'fear'
        ]
        
        # Count positive and negative words
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        # Calculate sentiment score
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        sentiment_score = (positive_count - negative_count) / max(1, total_words)
        return max(-1.0, min(1.0, sentiment_score * 10))  # Scale and bound
    
    def _extract_keywords(self, text: str, symbol: str) -> List[str]:
        """Extract relevant keywords from text"""
        text = text.lower()
        
        # Common trading/financial keywords
        financial_keywords = [
            'earnings', 'revenue', 'profit', 'dividend', 'split', 'buyback',
            'guidance', 'outlook', 'forecast', 'beat', 'miss', 'upgrade', 'downgrade',
            'analyst', 'target', 'rating', 'recommendation', 'valuation'
        ]
        
        # Symbol-specific keywords
        symbol_keywords = {
            'AAPL': ['iphone', 'ipad', 'mac', 'apple', 'ios', 'services'],
            'NVDA': ['gpu', 'ai', 'datacenter', 'gaming', 'nvidia', 'chips'],
            'MSFT': ['azure', 'office', 'windows', 'cloud', 'microsoft', 'teams'],
            'QQQ': ['nasdaq', 'tech', 'technology', 'etf', 'index'],
            'SPY': ['sp500', 's&p', 'market', 'index', 'etf', 'broad']
        }
        
        keywords = []
        
        # Check for financial keywords
        for keyword in financial_keywords:
            if keyword in text:
                keywords.append(keyword)
        
        # Check for symbol-specific keywords
        symbol_specific = symbol_keywords.get(symbol, [])
        for keyword in symbol_specific:
            if keyword in text:
                keywords.append(keyword)
        
        return keywords
    
    def _calculate_fear_greed_indicator(self, overall_sentiment: float, signals: List[SentimentSignal]) -> str:
        """Calculate fear/greed indicator"""
        # Extreme sentiment often indicates fear or greed
        extreme_signals = sum(1 for s in signals if abs(s.sentiment_score) > 60)
        high_volume_signals = sum(1 for s in signals if s.volume > 50)
        
        if overall_sentiment > 50 or extreme_signals > 2:
            return "EXTREME_GREED"
        elif overall_sentiment > 25:
            return "GREED"
        elif overall_sentiment > -25:
            return "NEUTRAL"
        elif overall_sentiment > -50:
            return "FEAR"
        else:
            return "EXTREME_FEAR"
    
    def _identify_contrarian_signals(self, signals: List[SentimentSignal], overall_sentiment: float) -> List[str]:
        """Identify contrarian signals"""
        contrarian_signals = []
        
        # Check for extreme sentiment (contrarian opportunity)
        if abs(overall_sentiment) > 60:
            contrarian_signals.append(f"Extreme sentiment detected: {overall_sentiment:.1f}")
        
        # Check individual signals for contrarian indicators
        for signal in signals:
            if signal.contrarian_indicator:
                contrarian_signals.append(f"{signal.source}: contrarian pattern detected")
        
        # Check for sentiment divergence from price action
        if len(signals) > 1:
            sentiment_directions = [s.trend_direction for s in signals]
            if sentiment_directions.count("UP") == sentiment_directions.count("DOWN"):
                contrarian_signals.append("Mixed sentiment signals - potential reversal")
        
        return contrarian_signals
    
    def _detect_contrarian_news(self, news_data: Dict, sentiment: float) -> bool:
        """Detect contrarian news patterns"""
        try:
            articles = news_data.get('articles', [])
            
            # Too much unanimous sentiment can be contrarian
            if len(articles) > 5 and abs(sentiment) > 0.8:
                return True
            
            # Check for extreme headlines
            extreme_headlines = sum(1 for article in articles 
                                  if any(word in article.get('headline', '').lower() 
                                       for word in ['crash', 'moon', 'explode', 'collapse', 'skyrocket']))
            
            return extreme_headlines > len(articles) * 0.3  # >30% extreme headlines
            
        except Exception as e:
            logger.error(f"Contrarian news detection error: {e}")
            return False
    
    def _detect_contrarian_social(self, social_data: Dict, sentiment: float) -> bool:
        """Detect contrarian social media patterns"""
        try:
            posts = social_data.get('posts', [])
            
            # High engagement with extreme sentiment can be contrarian
            high_engagement_posts = [p for p in posts if p.get('engagement', 0) > 100]
            
            if len(high_engagement_posts) > 5 and abs(sentiment) > 0.7:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Contrarian social detection error: {e}")
            return False
    
    def _update_trend_history(self, symbol: str, sentiment: float):
        """Update sentiment trend history"""
        if symbol not in self.trend_history:
            self.trend_history[symbol] = []
        
        self.trend_history[symbol].append({
            'sentiment': sentiment,
            'timestamp': datetime.now()
        })
        
        # Keep last 50 data points
        if len(self.trend_history[symbol]) > 50:
            self.trend_history[symbol].pop(0)
    
    def _calculate_overall_sentiment(self, signals: List[SentimentSignal]) -> Tuple[float, float, float]:
        """Calculate overall sentiment from individual signals"""
        if not signals:
            return 0.0, 0.0, 0.0
        
        weighted_sentiments = []
        total_weight = 0
        
        for signal in signals:
            source_weight = self.source_weights.get(signal.source, 0.1)
            confidence_weight = signal.confidence
            
            total_signal_weight = source_weight * confidence_weight
            weighted_sentiment = signal.sentiment_score * total_signal_weight
            
            weighted_sentiments.append(weighted_sentiment)
            total_weight += total_signal_weight
        
        overall_sentiment = sum(weighted_sentiments) / total_weight if total_weight > 0 else 0.0
        
        # Calculate confidence based on number of sources and their individual confidences
        avg_confidence = np.mean([s.confidence for s in signals])
        source_diversity = len(set(s.source for s in signals)) / 5  # Max 5 sources
        confidence = avg_confidence * source_diversity
        
        # Calculate trend momentum
        trend_momentum = self._calculate_trend_momentum(signals)
        
        return round(overall_sentiment, 1), round(confidence, 3), round(trend_momentum, 1)
    
    def _calculate_trend_momentum(self, signals: List[SentimentSignal]) -> float:
        """Calculate sentiment trend momentum"""
        try:
            # Look at recent vs older signals
            now = datetime.now()
            recent_signals = [s for s in signals if (now - s.timestamp).seconds < 3600]  # Last hour
            older_signals = [s for s in signals if (now - s.timestamp).seconds >= 3600]   # Older than hour
            
            if not recent_signals:
                return 0.0
            
            recent_sentiment = np.mean([s.sentiment_score for s in recent_signals])
            
            if older_signals:
                older_sentiment = np.mean([s.sentiment_score for s in older_signals])
                momentum = recent_sentiment - older_sentiment
            else:
                momentum = recent_sentiment * 0.1  # Reduced momentum if no comparison
            
            return max(-50, min(50, momentum))
            
        except Exception as e:
            logger.error(f"Trend momentum calculation error: {e}")
            return 0.0
    
    def _extract_key_themes(self, signals: List[SentimentSignal]) -> List[str]:
        """Extract key themes from sentiment signals"""
        try:
            all_keywords = []
            for signal in signals:
                all_keywords.extend(signal.keywords)
            
            # Count keyword frequency
            keyword_counts = {}
            for keyword in all_keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
            
            # Return top keywords as themes
            sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
            return [keyword for keyword, count in sorted_keywords[:5]]
            
        except Exception as e:
            logger.error(f"Theme extraction error: {e}")
            return []
    
    def _determine_retail_sentiment(self, signals: List[SentimentSignal]) -> str:
        """Determine overall retail sentiment"""
        retail_sources = ['reddit', 'twitter', 'social_general']
        retail_signals = [s for s in signals if s.source in retail_sources]
        
        if not retail_signals:
            return "NEUTRAL"
        
        avg_sentiment = np.mean([s.sentiment_score for s in retail_signals])
        
        if avg_sentiment > 30:
            return "VERY_BULLISH"
        elif avg_sentiment > 15:
            return "BULLISH"
        elif avg_sentiment > -15:
            return "NEUTRAL"
        elif avg_sentiment > -30:
            return "BEARISH"
        else:
            return "VERY_BEARISH"
    
    def _determine_institutional_sentiment(self, signals: List[SentimentSignal]) -> str:
        """Determine institutional sentiment"""
        institutional_signals = [s for s in signals if s.source in ['institutional', 'news']]
        
        if not institutional_signals:
            return "NEUTRAL"
        
        avg_sentiment = np.mean([s.sentiment_score for s in institutional_signals])
        
        if avg_sentiment > 20:
            return "BULLISH"
        elif avg_sentiment > -20:
            return "NEUTRAL"
        else:
            return "BEARISH"
    
    def _calculate_social_buzz_level(self, signals: List[SentimentSignal]) -> str:
        """Calculate social media buzz level"""
        total_volume = sum(s.volume for s in signals if s.source in ['reddit', 'twitter', 'social_general'])
        high_intensity_signals = sum(1 for s in signals if s.emotional_intensity == "HIGH")
        
        if total_volume > 100 or high_intensity_signals > 2:
            return "HIGH"
        elif total_volume > 50 or high_intensity_signals > 0:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_fallback_sentiment(self, symbol: str) -> SentimentAnalysis:
        """Generate fallback sentiment analysis"""
        fallback_signal = SentimentSignal(
            source="fallback",
            sentiment_score=0.0,
            confidence=0.5,
            volume=0,
            keywords=[],
            timestamp=datetime.now(),
            trend_direction="NEUTRAL",
            emotional_intensity="LOW"
        )
        
        return SentimentAnalysis(
            overall_sentiment=0.0,
            confidence=0.5,
            trend_momentum=0.0,
            signals=[fallback_signal],
            key_themes=[],
            retail_sentiment="NEUTRAL",
            institutional_sentiment="NEUTRAL",
            social_buzz_level="LOW",
            fear_greed_indicator="NEUTRAL",
            contrarian_signals=[]
        )

# Supporting simulator classes for sentiment data
class NewsSimulator:
    """Simulate news sentiment data"""
    
    def __init__(self):
        self.news_cache = {}
        logger.info("📰 News Simulator initialized")
    
    async def get_news_sentiment(self, symbol: str, quote_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate simulated news sentiment"""
        try:
            change_percent = float(quote_data.get('change_percent', 0))
            
            # Generate articles based on price movement
            articles = []
            num_articles = random.randint(3, 12)
            
            for i in range(num_articles):
                # Sentiment correlation with price movement
                if abs(change_percent) > 3:
                    sentiment_bias = 1 if change_percent > 0 else -1
                else:
                    sentiment_bias = random.choice([-1, 0, 1])
                
                headline = self._generate_headline(symbol, sentiment_bias)
                
                articles.append({
                    'headline': headline,
                    'summary': f"Analysis of {symbol} market movement and outlook.",
                    'relevance_score': random.uniform(0.7, 1.0),
                    'hours_old': random.randint(1, 24),
                    'sentiment_bias': sentiment_bias
                })
            
            return {'articles': articles}
            
        except Exception as e:
            logger.error(f"News simulation error: {e}")
            return {'articles': []}
    
    def _generate_headline(self, symbol: str, sentiment_bias: int) -> str:
        """Generate realistic news headlines"""
        positive_templates = [
            f"{symbol} Surges on Strong Earnings Beat",
            f"Analysts Upgrade {symbol} Price Target",
            f"{symbol} Reports Record Revenue Growth",
            f"Bullish Outlook for {symbol} This Quarter"
        ]
        
        negative_templates = [
            f"{symbol} Falls on Disappointing Guidance",
            f"Concerns Mount Over {symbol} Valuation",
            f"{symbol} Faces Regulatory Headwinds",
            f"Analysts Lower {symbol} Expectations"
        ]
        
        neutral_templates = [
            f"{symbol} Trading Sideways Ahead of Earnings",
            f"Mixed Signals for {symbol} Investors",
            f"{symbol} Maintains Steady Performance",
            f"Market Watches {symbol} for Direction"
        ]
        
        if sentiment_bias > 0:
            return random.choice(positive_templates)
        elif sentiment_bias < 0:
            return random.choice(negative_templates)
        else:
            return random.choice(neutral_templates)

class SocialMediaSimulator:
    """Simulate general social media sentiment"""
    
    def __init__(self):
        logger.info("📱 Social Media Simulator initialized")
    
    async def get_social_sentiment(self, symbol: str, quote_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate simulated social media posts"""
        try:
            change_percent = float(quote_data.get('change_percent', 0))
            
            posts = []
            num_posts = random.randint(10, 30)
            
            for i in range(num_posts):
                sentiment_direction = self._determine_post_sentiment(change_percent)
                text = self._generate_post_text(symbol, sentiment_direction)
                
                posts.append({
                    'text': text,
                    'engagement': random.randint(1, 500),
                    'platform': random.choice(['general', 'forums', 'blogs']),
                    'sentiment_direction': sentiment_direction
                })
            
            return {'posts': posts}
            
        except Exception as e:
            logger.error(f"Social media simulation error: {e}")
            return {'posts': []}
    
    def _determine_post_sentiment(self, change_percent: float) -> str:
        """Determine post sentiment based on price movement"""
        if change_percent > 2:
            return random.choices(['positive', 'neutral'], weights=[0.7, 0.3])[0]
        elif change_percent < -2:
            return random.choices(['negative', 'neutral'], weights=[0.7, 0.3])[0]
        else:
            return random.choices(['positive', 'neutral', 'negative'], weights=[0.3, 0.4, 0.3])[0]
    
    def _generate_post_text(self, symbol: str, sentiment: str) -> str:
        """Generate social media post text"""
        if sentiment == 'positive':
            templates = [
                f"{symbol} looking strong today!",
                f"Bullish on {symbol} long term",
                f"{symbol} breakout incoming?",
                f"Love the momentum on {symbol}"
            ]
        elif sentiment == 'negative':
            templates = [
                f"{symbol} taking a hit today",
                f"Concerned about {symbol} outlook",
                f"{symbol} looks weak here",
                f"Time to take profits on {symbol}?"
            ]
        else:
            templates = [
                f"{symbol} trading sideways",
                f"Watching {symbol} for direction",
                f"{symbol} holding support levels",
                f"Mixed signals on {symbol}"
            ]
        
        return random.choice(templates)

class RedditSimulator:
    """Simulate Reddit sentiment data"""
    
    def __init__(self):
        logger.info("🤖 Reddit Simulator initialized")
    
    async def get_reddit_sentiment(self, symbol: str, quote_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate simulated Reddit posts"""
        try:
            change_percent = float(quote_data.get('change_percent', 0))
            
            posts = []
            num_posts = random.randint(5, 20)
            
            for i in range(num_posts):
                # Reddit tends to be more extreme and meme-driven
                if abs(change_percent) > 1:
                    sentiment_intensity = random.choice(['high', 'medium'])
                else:
                    sentiment_intensity = random.choice(['medium', 'low'])
                
                text = self._generate_reddit_text(symbol, change_percent, sentiment_intensity)
                
                posts.append({
                    'text': text,
                    'engagement': random.randint(10, 1000),  # Higher engagement on Reddit
                    'subreddit': random.choice(['wallstreetbets', 'stocks', 'investing']),
                    'intensity': sentiment_intensity
                })
            
            return {'posts': posts}
            
        except Exception as e:
            logger.error(f"Reddit simulation error: {e}")
            return {'posts': []}
    
    def _generate_reddit_text(self, symbol: str, change_percent: float, intensity: str) -> str:
        """Generate Reddit-style post text"""
        if change_percent > 1 and intensity == 'high':
            templates = [
                f"{symbol} to the moon! 🚀🚀🚀",
                f"YOLO all in on {symbol}",
                f"{symbol} printing money today",
                f"Diamond hands on {symbol} 💎🙌"
            ]
        elif change_percent < -1 and intensity == 'high':
            templates = [
                f"{symbol} drilling to earth's core",
                f"My {symbol} calls are dead",
                f"{symbol} rug pull in progress",
                f"Stop the count on {symbol}"
            ]
        else:
            templates = [
                f"What's everyone's thoughts on {symbol}?",
                f"{symbol} DD needed",
                f"Holding {symbol} through this",
                f"{symbol} chart looking interesting"
            ]
        
        return random.choice(templates)

class TwitterSimulator:
    """Simulate Twitter sentiment data"""
    
    def __init__(self):
        logger.info("🐦 Twitter Simulator initialized")
    
    async def get_twitter_sentiment(self, symbol: str, quote_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate simulated Twitter posts"""
        try:
            change_percent = float(quote_data.get('change_percent', 0))
            
            posts = []
            num_posts = random.randint(8, 25)
            
            for i in range(num_posts):
                # Twitter sentiment often follows price movement
                sentiment_correlation = random.uniform(0.6, 0.9)  # High correlation
                
                if change_percent > 0 and random.random() < sentiment_correlation:
                    sentiment_direction = 'positive'
                elif change_percent < 0 and random.random() < sentiment_correlation:
                    sentiment_direction = 'negative'
                else:
                    sentiment_direction = random.choice(['positive', 'neutral', 'negative'])
                
                text = self._generate_twitter_text(symbol, sentiment_direction)
                
                posts.append({
                    'text': text,
                    'engagement': random.randint(5, 200),
                    'retweets': random.randint(0, 50),
                    'sentiment_direction': sentiment_direction
                })
            
            return {'posts': posts}
            
        except Exception as e:
            logger.error(f"Twitter simulation error: {e}")
            return {'posts': []}
    
    def _generate_twitter_text(self, symbol: str, sentiment: str) -> str:
        """Generate Twitter-style post text"""
        if sentiment == 'positive':
            templates = [
                f"${symbol} looking bullish 📈",
                f"Buying the dip on ${symbol}",
                f"${symbol} breakout confirmed ✅",
                f"${symbol} strength continues"
            ]
        elif sentiment == 'negative':
            templates = [
                f"${symbol} weakness showing 📉",
                f"Selling ${symbol} on this bounce",
                f"${symbol} breakdown in progress",
                f"Avoiding ${symbol} for now"
            ]
        else:
            templates = [
                f"${symbol} consolidating here",
                f"Watching ${symbol} for entry",
                f"${symbol} at key levels",
                f"${symbol} decision time"
            ]
        
        return random.choice(templates)

# Export main classes
__all__ = [
    'AdvancedSentimentAnalyzer', 'SentimentAnalysis', 'SentimentSignal',
    'NewsSimulator', 'SocialMediaSimulator', 'RedditSimulator', 'TwitterSimulator'
]

logger.info("💭 Advanced Sentiment Analysis module loaded successfully")
logger.info("📊 Multi-source sentiment fusion with NLP processing enabled")
logger.info("🎯 Contrarian signal detection and retail vs institutional analysis active")
