"""
Advanced Search Engine
Comprehensive search capabilities with filtering, ranking, and analytics
"""

import asyncio
import logging
import time
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, Counter
import math

from app.core.enhanced_cache import enhanced_cache
from app.core.database import engine
from sqlalchemy import text

logger = logging.getLogger(__name__)


class SearchType(Enum):
    """Search types"""
    FULL_TEXT = "full_text"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"
    FILTERED = "filtered"
    AGGREGATED = "aggregated"


class SortOrder(Enum):
    """Sort orders"""
    RELEVANCE = "relevance"
    DATE_ASC = "date_asc"
    DATE_DESC = "date_desc"
    PRICE_ASC = "price_asc"
    PRICE_DESC = "price_desc"
    POPULARITY = "popularity"
    RATING = "rating"


@dataclass
class SearchQuery:
    """Search query data structure"""
    query_id: str
    query_text: str
    search_type: SearchType
    filters: Dict[str, Any]
    sort_order: SortOrder
    limit: int
    offset: int
    user_id: Optional[str]
    timestamp: float


@dataclass
class SearchResult:
    """Search result data structure"""
    result_id: str
    content_type: str
    content_id: str
    title: str
    description: str
    relevance_score: float
    metadata: Dict[str, Any]
    timestamp: float


@dataclass
class SearchAnalytics:
    """Search analytics data structure"""
    query_id: str
    query_text: str
    result_count: int
    click_through_rate: float
    average_position: float
    search_time_ms: float
    user_id: Optional[str]
    timestamp: float


class SearchEngine:
    """Advanced search engine with filtering and ranking"""
    
    def __init__(self):
        self.search_index = defaultdict(list)
        self.search_history = defaultdict(list)
        self.search_analytics = {}
        self.filter_definitions = {}
        self.ranking_weights = {
            "relevance": 0.4,
            "popularity": 0.3,
            "recency": 0.2,
            "user_preference": 0.1
        }
        
        # Configuration
        self.max_results = 1000
        self.default_limit = 20
        self.fuzzy_threshold = 0.7
        self.analytics_retention = 30 * 24 * 3600  # 30 days
        
        # Statistics
        self.stats = {
            "searches_performed": 0,
            "results_returned": 0,
            "filters_applied": 0,
            "analytics_collected": 0
        }
        
        # Start background tasks
        asyncio.create_task(self._index_build_loop())
        asyncio.create_task(self._analytics_processing_loop())
    
    async def start_search_engine(self):
        """Start the search engine"""
        logger.info("Starting advanced search engine")
        
        # Initialize filter definitions
        await self._initialize_filters()
        
        # Build initial index
        await self._build_search_index()
        
        logger.info("Advanced search engine started")
    
    async def _initialize_filters(self):
        """Initialize filter definitions"""
        try:
            self.filter_definitions = {
                "market_status": {
                    "type": "enum",
                    "values": ["active", "inactive", "pending", "closed"],
                    "description": "Market status filter"
                },
                "price_range": {
                    "type": "range",
                    "min": 0,
                    "max": 1000000,
                    "description": "Price range filter"
                },
                "date_range": {
                    "type": "date_range",
                    "description": "Date range filter"
                },
                "category": {
                    "type": "enum",
                    "values": ["politics", "sports", "entertainment", "technology", "finance"],
                    "description": "Category filter"
                },
                "user_rating": {
                    "type": "range",
                    "min": 0,
                    "max": 5,
                    "description": "User rating filter"
                },
                "volume": {
                    "type": "range",
                    "min": 0,
                    "max": 1000000,
                    "description": "Trading volume filter"
                }
            }
            
            logger.info("Filter definitions initialized")
            
        except Exception as e:
            logger.error(f"Error initializing filters: {e}")
    
    async def _build_search_index(self):
        """Build search index from database"""
        try:
            # Index markets
            await self._index_markets()
            
            # Index users
            await self._index_users()
            
            # Index trades
            await self._index_trades()
            
            logger.info("Search index built successfully")
            
        except Exception as e:
            logger.error(f"Error building search index: {e}")
    
    async def _index_markets(self):
        """Index market data"""
        try:
            with engine.connect() as conn:
                query = """
                SELECT market_id, title, description, category, status, 
                       current_price, volume, created_at, updated_at
                FROM markets
                """
                result = conn.execute(text(query))
                
                for row in result:
                    market_data = {
                        "content_type": "market",
                        "content_id": str(row[0]),
                        "title": row[1],
                        "description": row[2],
                        "category": row[3],
                        "status": row[4],
                        "current_price": float(row[5]) if row[5] else 0,
                        "volume": float(row[6]) if row[6] else 0,
                        "created_at": row[7].timestamp() if row[7] else 0,
                        "updated_at": row[8].timestamp() if row[8] else 0
                    }
                    
                    # Add to search index
                    self._add_to_index(market_data)
                
        except Exception as e:
            logger.error(f"Error indexing markets: {e}")
    
    async def _index_users(self):
        """Index user data"""
        try:
            with engine.connect() as conn:
                query = """
                SELECT user_id, username, email, created_at, last_login
                FROM users
                """
                result = conn.execute(text(query))
                
                for row in result:
                    user_data = {
                        "content_type": "user",
                        "content_id": str(row[0]),
                        "title": row[1],
                        "description": f"User: {row[1]}",
                        "email": row[2],
                        "created_at": row[3].timestamp() if row[3] else 0,
                        "last_login": row[4].timestamp() if row[4] else 0
                    }
                    
                    # Add to search index
                    self._add_to_index(user_data)
                
        except Exception as e:
            logger.error(f"Error indexing users: {e}")
    
    async def _index_trades(self):
        """Index trade data"""
        try:
            with engine.connect() as conn:
                query = """
                SELECT trade_id, market_id, user_id, amount, price, 
                       trade_type, timestamp
                FROM trades
                """
                result = conn.execute(text(query))
                
                for row in result:
                    trade_data = {
                        "content_type": "trade",
                        "content_id": str(row[0]),
                        "title": f"Trade {row[0]}",
                        "description": f"Trade of {row[3]} at {row[4]}",
                        "market_id": str(row[1]),
                        "user_id": str(row[2]),
                        "amount": float(row[3]) if row[3] else 0,
                        "price": float(row[4]) if row[4] else 0,
                        "trade_type": row[5],
                        "timestamp": row[6].timestamp() if row[6] else 0
                    }
                    
                    # Add to search index
                    self._add_to_index(trade_data)
                
        except Exception as e:
            logger.error(f"Error indexing trades: {e}")
    
    def _add_to_index(self, data: Dict[str, Any]):
        """Add data to search index"""
        try:
            # Extract searchable text
            searchable_text = f"{data.get('title', '')} {data.get('description', '')}"
            
            # Tokenize text
            tokens = self._tokenize_text(searchable_text)
            
            # Add to index
            for token in tokens:
                self.search_index[token].append(data)
            
        except Exception as e:
            logger.error(f"Error adding to index: {e}")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for indexing"""
        try:
            # Convert to lowercase and split
            tokens = re.findall(r'\b\w+\b', text.lower())
            
            # Remove stop words (simplified)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return []
    
    async def search(self, query_text: str, search_type: SearchType = SearchType.FULL_TEXT,
                    filters: Dict[str, Any] = None, sort_order: SortOrder = SortOrder.RELEVANCE,
                    limit: int = None, offset: int = 0, user_id: str = None) -> Dict[str, Any]:
        """Perform a search"""
        try:
            start_time = time.time()
            
            # Create search query
            query = SearchQuery(
                query_id=f"search_{int(time.time())}_{hash(query_text)}",
                query_text=query_text,
                search_type=search_type,
                filters=filters or {},
                sort_order=sort_order,
                limit=limit or self.default_limit,
                offset=offset,
                user_id=user_id,
                timestamp=time.time()
            )
            
            # Perform search based on type
            if search_type == SearchType.FULL_TEXT:
                results = await self._full_text_search(query)
            elif search_type == SearchType.FUZZY:
                results = await self._fuzzy_search(query)
            elif search_type == SearchType.FILTERED:
                results = await self._filtered_search(query)
            else:
                results = await self._full_text_search(query)
            
            # Apply filters
            if filters:
                results = await self._apply_filters(results, filters)
                self.stats["filters_applied"] += 1
            
            # Sort results
            results = await self._sort_results(results, sort_order)
            
            # Apply pagination
            total_results = len(results)
            results = results[offset:offset + query.limit]
            
            # Calculate search time
            search_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Store search analytics
            await self._store_search_analytics(query, total_results, search_time)
            
            # Update statistics
            self.stats["searches_performed"] += 1
            self.stats["results_returned"] += len(results)
            
            return {
                "query": query_text,
                "search_type": search_type.value,
                "filters": filters,
                "sort_order": sort_order.value,
                "results": [{
                    "result_id": r.result_id,
                    "content_type": r.content_type,
                    "content_id": r.content_id,
                    "title": r.title,
                    "description": r.description,
                    "relevance_score": r.relevance_score,
                    "metadata": r.metadata
                } for r in results],
                "pagination": {
                    "total_results": total_results,
                    "limit": query.limit,
                    "offset": offset,
                    "has_more": offset + len(results) < total_results
                },
                "search_time_ms": search_time,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return {
                "query": query_text,
                "results": [],
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _full_text_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform full text search"""
        try:
            # Tokenize query
            query_tokens = self._tokenize_text(query.query_text)
            
            # Find matching documents
            matching_docs = defaultdict(int)
            for token in query_tokens:
                if token in self.search_index:
                    for doc in self.search_index[token]:
                        doc_key = f"{doc['content_type']}_{doc['content_id']}"
                        matching_docs[doc_key] += 1
            
            # Convert to search results
            results = []
            for doc_key, score in matching_docs.items():
                # Find the document
                doc = None
                for token in query_tokens:
                    if token in self.search_index:
                        for d in self.search_index[token]:
                            if f"{d['content_type']}_{d['content_id']}" == doc_key:
                                doc = d
                                break
                        if doc:
                            break
                
                if doc:
                    # Calculate relevance score
                    relevance_score = score / len(query_tokens)
                    
                    result = SearchResult(
                        result_id=f"result_{doc_key}_{int(time.time())}",
                        content_type=doc["content_type"],
                        content_id=doc["content_id"],
                        title=doc.get("title", ""),
                        description=doc.get("description", ""),
                        relevance_score=relevance_score,
                        metadata=doc,
                        timestamp=time.time()
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in full text search: {e}")
            return []
    
    async def _fuzzy_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform fuzzy search"""
        try:
            query_tokens = self._tokenize_text(query.query_text)
            matching_docs = defaultdict(float)
            
            for token in query_tokens:
                for index_token in self.search_index.keys():
                    # Calculate similarity
                    similarity = self._calculate_similarity(token, index_token)
                    
                    if similarity >= self.fuzzy_threshold:
                        for doc in self.search_index[index_token]:
                            doc_key = f"{doc['content_type']}_{doc['content_id']}"
                            matching_docs[doc_key] = max(matching_docs[doc_key], similarity)
            
            # Convert to search results
            results = []
            for doc_key, score in matching_docs.items():
                # Find the document
                doc = None
                for index_token in self.search_index.keys():
                    for d in self.search_index[index_token]:
                        if f"{d['content_type']}_{d['content_id']}" == doc_key:
                            doc = d
                            break
                    if doc:
                        break
                
                if doc:
                    result = SearchResult(
                        result_id=f"result_{doc_key}_{int(time.time())}",
                        content_type=doc["content_type"],
                        content_id=doc["content_id"],
                        title=doc.get("title", ""),
                        description=doc.get("description", ""),
                        relevance_score=score,
                        metadata=doc,
                        timestamp=time.time()
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in fuzzy search: {e}")
            return []
    
    async def _filtered_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform filtered search"""
        try:
            # Start with full text search
            results = await self._full_text_search(query)
            
            # Apply filters
            if query.filters:
                results = await self._apply_filters(results, query.filters)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in filtered search: {e}")
            return []
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using Levenshtein distance"""
        try:
            if len(str1) < len(str2):
                str1, str2 = str2, str1
            
            if len(str2) == 0:
                return 0.0
            
            # Calculate Levenshtein distance
            distance = self._levenshtein_distance(str1, str2)
            max_len = max(len(str1), len(str2))
            
            return 1.0 - (distance / max_len)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def _levenshtein_distance(self, str1: str, str2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        try:
            if len(str1) < len(str2):
                str1, str2 = str2, str1
            
            if len(str2) == 0:
                return len(str1)
            
            previous_row = list(range(len(str2) + 1))
            for i, c1 in enumerate(str1):
                current_row = [i + 1]
                for j, c2 in enumerate(str2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
            
        except Exception as e:
            logger.error(f"Error calculating Levenshtein distance: {e}")
            return 0
    
    async def _apply_filters(self, results: List[SearchResult], filters: Dict[str, Any]) -> List[SearchResult]:
        """Apply filters to search results"""
        try:
            filtered_results = []
            
            for result in results:
                metadata = result.metadata
                include_result = True
                
                for filter_name, filter_value in filters.items():
                    if not self._apply_filter(result, filter_name, filter_value):
                        include_result = False
                        break
                
                if include_result:
                    filtered_results.append(result)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return results
    
    def _apply_filter(self, result: SearchResult, filter_name: str, filter_value: Any) -> bool:
        """Apply a single filter to a result"""
        try:
            metadata = result.metadata
            
            if filter_name == "market_status":
                return metadata.get("status") == filter_value
            
            elif filter_name == "price_range":
                if isinstance(filter_value, dict):
                    min_price = filter_value.get("min", 0)
                    max_price = filter_value.get("max", float('inf'))
                    price = metadata.get("current_price", 0)
                    return min_price <= price <= max_price
                return True
            
            elif filter_name == "date_range":
                if isinstance(filter_value, dict):
                    start_date = filter_value.get("start", 0)
                    end_date = filter_value.get("end", float('inf'))
                    created_at = metadata.get("created_at", 0)
                    return start_date <= created_at <= end_date
                return True
            
            elif filter_name == "category":
                return metadata.get("category") == filter_value
            
            elif filter_name == "user_rating":
                if isinstance(filter_value, dict):
                    min_rating = filter_value.get("min", 0)
                    max_rating = filter_value.get("max", 5)
                    rating = metadata.get("rating", 0)
                    return min_rating <= rating <= max_rating
                return True
            
            elif filter_name == "volume":
                if isinstance(filter_value, dict):
                    min_volume = filter_value.get("min", 0)
                    max_volume = filter_value.get("max", float('inf'))
                    volume = metadata.get("volume", 0)
                    return min_volume <= volume <= max_volume
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying filter {filter_name}: {e}")
            return True
    
    async def _sort_results(self, results: List[SearchResult], sort_order: SortOrder) -> List[SearchResult]:
        """Sort search results"""
        try:
            if sort_order == SortOrder.RELEVANCE:
                return sorted(results, key=lambda x: x.relevance_score, reverse=True)
            
            elif sort_order == SortOrder.DATE_ASC:
                return sorted(results, key=lambda x: x.metadata.get("created_at", 0))
            
            elif sort_order == SortOrder.DATE_DESC:
                return sorted(results, key=lambda x: x.metadata.get("created_at", 0), reverse=True)
            
            elif sort_order == SortOrder.PRICE_ASC:
                return sorted(results, key=lambda x: x.metadata.get("current_price", 0))
            
            elif sort_order == SortOrder.PRICE_DESC:
                return sorted(results, key=lambda x: x.metadata.get("current_price", 0), reverse=True)
            
            elif sort_order == SortOrder.POPULARITY:
                return sorted(results, key=lambda x: x.metadata.get("volume", 0), reverse=True)
            
            elif sort_order == SortOrder.RATING:
                return sorted(results, key=lambda x: x.metadata.get("rating", 0), reverse=True)
            
            else:
                return results
                
        except Exception as e:
            logger.error(f"Error sorting results: {e}")
            return results
    
    async def _store_search_analytics(self, query: SearchQuery, result_count: int, search_time: float):
        """Store search analytics"""
        try:
            analytics = SearchAnalytics(
                query_id=query.query_id,
                query_text=query.query_text,
                result_count=result_count,
                click_through_rate=0.0,  # Will be updated when user clicks
                average_position=0.0,    # Will be calculated
                search_time_ms=search_time,
                user_id=query.user_id,
                timestamp=query.timestamp
            )
            
            # Store in cache
            await enhanced_cache.set(
                f"search_analytics_{query.query_id}",
                asdict(analytics),
                ttl=self.analytics_retention,
                tags=["search_analytics", query.user_id or "anonymous"]
            )
            
            # Store in search history
            if query.user_id:
                self.search_history[query.user_id].append(analytics)
            
            self.stats["analytics_collected"] += 1
            
        except Exception as e:
            logger.error(f"Error storing search analytics: {e}")
    
    async def _index_build_loop(self):
        """Background task to rebuild search index"""
        while True:
            try:
                await asyncio.sleep(3600)  # Rebuild index every hour
                await self._build_search_index()
            except Exception as e:
                logger.error(f"Error in index build loop: {e}")
                await asyncio.sleep(3600)
    
    async def _analytics_processing_loop(self):
        """Background task to process search analytics"""
        while True:
            try:
                await asyncio.sleep(1800)  # Process analytics every 30 minutes
                await self._process_search_analytics()
            except Exception as e:
                logger.error(f"Error in analytics processing loop: {e}")
                await asyncio.sleep(1800)
    
    async def _process_search_analytics(self):
        """Process search analytics"""
        try:
            # Calculate popular search terms
            popular_terms = self._calculate_popular_search_terms()
            
            # Calculate search performance metrics
            performance_metrics = self._calculate_search_performance()
            
            # Store analytics
            await enhanced_cache.set(
                "search_analytics_summary",
                {
                    "popular_terms": popular_terms,
                    "performance_metrics": performance_metrics,
                    "timestamp": time.time()
                },
                ttl=3600,
                tags=["search_analytics", "summary"]
            )
            
        except Exception as e:
            logger.error(f"Error processing search analytics: {e}")
    
    def _calculate_popular_search_terms(self) -> List[Dict[str, Any]]:
        """Calculate popular search terms"""
        try:
            term_counts = Counter()
            
            for user_history in self.search_history.values():
                for analytics in user_history:
                    terms = self._tokenize_text(analytics.query_text)
                    for term in terms:
                        term_counts[term] += 1
            
            return [
                {"term": term, "count": count}
                for term, count in term_counts.most_common(20)
            ]
            
        except Exception as e:
            logger.error(f"Error calculating popular search terms: {e}")
            return []
    
    def _calculate_search_performance(self) -> Dict[str, Any]:
        """Calculate search performance metrics"""
        try:
            total_searches = 0
            total_results = 0
            total_search_time = 0
            
            for user_history in self.search_history.values():
                for analytics in user_history:
                    total_searches += 1
                    total_results += analytics.result_count
                    total_search_time += analytics.search_time_ms
            
            return {
                "total_searches": total_searches,
                "average_results_per_search": total_results / max(total_searches, 1),
                "average_search_time_ms": total_search_time / max(total_searches, 1),
                "search_success_rate": 1.0  # Simplified
            }
            
        except Exception as e:
            logger.error(f"Error calculating search performance: {e}")
            return {}
    
    async def get_search_suggestions(self, query_text: str, limit: int = 10) -> List[str]:
        """Get search suggestions based on query text"""
        try:
            if len(query_text) < 2:
                return []
            
            suggestions = []
            query_lower = query_text.lower()
            
            # Get suggestions from popular terms
            popular_terms = await enhanced_cache.get("search_analytics_summary")
            if popular_terms and "popular_terms" in popular_terms:
                for term_data in popular_terms["popular_terms"]:
                    term = term_data["term"]
                    if query_lower in term.lower() and term not in suggestions:
                        suggestions.append(term)
                        if len(suggestions) >= limit:
                            break
            
            # Get suggestions from index
            for index_term in self.search_index.keys():
                if query_lower in index_term.lower() and index_term not in suggestions:
                    suggestions.append(index_term)
                    if len(suggestions) >= limit:
                        break
            
            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Error getting search suggestions: {e}")
            return []
    
    async def get_search_analytics(self, user_id: Optional[str] = None, time_range: int = 86400) -> Dict[str, Any]:
        """Get search analytics"""
        try:
            cutoff_time = time.time() - time_range
            
            if user_id:
                user_history = self.search_history.get(user_id, [])
                recent_searches = [s for s in user_history if s.timestamp >= cutoff_time]
            else:
                recent_searches = []
                for user_history in self.search_history.values():
                    recent_searches.extend([s for s in user_history if s.timestamp >= cutoff_time])
            
            # Calculate analytics
            total_searches = len(recent_searches)
            unique_queries = len(set(s.query_text for s in recent_searches))
            average_results = sum(s.result_count for s in recent_searches) / max(total_searches, 1)
            average_search_time = sum(s.search_time_ms for s in recent_searches) / max(total_searches, 1)
            
            return {
                "total_searches": total_searches,
                "unique_queries": unique_queries,
                "average_results_per_search": average_results,
                "average_search_time_ms": average_search_time,
                "time_range": time_range,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting search analytics: {e}")
            return {}
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        return {
            "searches_performed": self.stats["searches_performed"],
            "results_returned": self.stats["results_returned"],
            "filters_applied": self.stats["filters_applied"],
            "analytics_collected": self.stats["analytics_collected"],
            "index_size": len(self.search_index),
            "total_documents": sum(len(docs) for docs in self.search_index.values()),
            "active_users": len(self.search_history),
            "filter_definitions": len(self.filter_definitions)
        }


# Global search engine instance
search_engine = SearchEngine()
