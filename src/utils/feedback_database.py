import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import os
from collections import defaultdict, Counter
from .utils import safe_mean, safe_divide, extract_keywords
from .paths import FEEDBACK_DB

def sanitize_for_json(obj):
    """Sanitize data for JSON serialization by converting numpy types to Python types."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
        return 0.0
    else:
        return obj

class AdvancedAnalytics:
    """Advanced analytics engine for enhanced monitoring."""

    def __init__(self, feedback_db):
        self.feedback_db = feedback_db

    def analyze_query_patterns(self, days: int = 30) -> Dict[str, Any]:
        """Deep query pattern analysis."""
        conn = self.feedback_db.get_connection()
        
        # Get query data
        query = '''
        SELECT query_text, user_rating, processing_time, chunks_used,
               query_strategy, timestamp, feedback_text
        FROM query_feedback
        WHERE timestamp >= datetime('now', '-{} days')
        '''.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return self._empty_analytics()

        # Pattern analysis
        patterns = {
            'total_queries': len(df),
            'unique_queries': df['query_text'].nunique(),
            'strategy_distribution': df['query_strategy'].value_counts().to_dict(),
            'avg_processing_time_by_strategy': df.groupby('query_strategy')['processing_time'].mean().to_dict(),
            'rating_by_strategy': df.groupby('query_strategy')['user_rating'].mean().to_dict(),
            'temporal_patterns': self._analyze_temporal_patterns(df),
            'query_complexity_trends': self._analyze_complexity_trends(df),
            'common_keywords': self._extract_common_keywords(df),
            'performance_trends': self._calculate_performance_trends(df)
        }

        return sanitize_for_json(patterns)

    def analyze_user_journey(self, session_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Analyze user journey patterns."""
        conn = self.feedback_db.get_connection()
        query = '''
        SELECT session_id, query_text, user_rating, timestamp, query_strategy
        FROM query_feedback
        WHERE session_id IS NOT NULL AND session_id != 'anonymous'
        ORDER BY session_id, timestamp
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return {'session_analysis': {}, 'journey_patterns': {}}

        # Group by session
        sessions = df.groupby('session_id')
        
        journey_analysis = {
            'avg_queries_per_session': sessions.size().mean(),
            'session_duration_avg': self._calculate_avg_session_duration(sessions),
            'strategy_transitions': self._analyze_strategy_transitions(sessions),
            'satisfaction_journey': self._analyze_satisfaction_journey(sessions),
            'common_query_sequences': self._find_common_sequences(sessions)
        }

        return sanitize_for_json({
            'session_analysis': journey_analysis,
            'total_sessions': len(sessions),
            'active_sessions_today': self._count_active_sessions_today(df)
        })

    def generate_performance_insights(self) -> Dict[str, Any]:
        """Generate AI-powered insights about system performance."""
        patterns = self.analyze_query_patterns()
        journey = self.analyze_user_journey()
        
        insights = {
            'performance_summary': self._generate_performance_summary(patterns),
            'optimization_suggestions': self._generate_optimization_suggestions(patterns, journey),
            'anomaly_detection': self._detect_anomalies(patterns),
            'trend_analysis': self._analyze_trends(patterns),
            'user_satisfaction_insights': self._analyze_satisfaction_insights(patterns, journey)
        }

        return sanitize_for_json(insights)

    def _empty_analytics(self) -> Dict[str, Any]:
        """Return empty analytics structure."""
        return {
            'total_queries': 0,
            'strategy_distribution': {},
            'temporal_patterns': {},
            'common_keywords': [],
            'performance_trends': {}
        }

    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal usage patterns."""
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        return {
            'peak_hours': df['hour'].value_counts().head(3).to_dict(),
            'peak_days': df['day_of_week'].value_counts().head(3).to_dict(),
            'hourly_distribution': df['hour'].value_counts().sort_index().to_dict()
        }

    def _analyze_complexity_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze query complexity trends."""
        df['query_length'] = df['query_text'].str.len()
        df['word_count'] = df['query_text'].str.split().str.len()
        
        return {
            'avg_query_length': df['query_length'].mean(),
            'avg_word_count': df['word_count'].mean(),
            'length_vs_rating_correlation': df['query_length'].corr(df['user_rating']),
            'complexity_distribution': {
                'short': len(df[df['word_count'] <= 5]),
                'medium': len(df[(df['word_count'] > 5) & (df['word_count'] <= 10)]),
                'long': len(df[df['word_count'] > 10])
            }
        }

    def _extract_common_keywords(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract and analyze common keywords."""
        all_keywords = []
        for query in df['query_text'].dropna():
            all_keywords.extend(extract_keywords(query))
        
        keyword_counts = Counter(all_keywords)
        return [
            {'keyword': word, 'count': count, 'frequency': count/len(df)}
            for word, count in keyword_counts.most_common(10)
        ]

    def _calculate_performance_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate performance trends over time."""
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_stats = df.groupby('date').agg({
            'user_rating': 'mean',
            'processing_time': 'mean',
            'query_text': 'count'
        }).rename(columns={'query_text': 'query_count'})

        return {
            'daily_ratings': daily_stats['user_rating'].to_dict(),
            'daily_response_times': daily_stats['processing_time'].to_dict(),
            'daily_query_counts': daily_stats['query_count'].to_dict(),
            'rating_trend': 'improving' if daily_stats['user_rating'].is_monotonic_increasing else 'declining'
        }

    def _calculate_avg_session_duration(self, sessions) -> float:
        """Calculate average session duration."""
        durations = []
        for session_id, group in sessions:
            if len(group) > 1:
                start_time = pd.to_datetime(group['timestamp'].min())
                end_time = pd.to_datetime(group['timestamp'].max())
                duration = (end_time - start_time).total_seconds() / 60  # minutes
                durations.append(duration)
        return np.mean(durations) if durations else 0.0

    def _analyze_strategy_transitions(self, sessions) -> Dict[str, Any]:
        """Analyze how users transition between query strategies."""
        transitions = defaultdict(int)
        for session_id, group in sessions:
            strategies = group['query_strategy'].tolist()
            for i in range(len(strategies) - 1):
                transition = f"{strategies[i]} -> {strategies[i+1]}"
                transitions[transition] += 1
        return dict(transitions)

    def _analyze_satisfaction_journey(self, sessions) -> Dict[str, Any]:
        """Analyze how satisfaction changes within sessions."""
        satisfaction_patterns = {
            'improving': 0,
            'declining': 0,
            'stable': 0
        }

        for session_id, group in sessions:
            if len(group) >= 2:
                ratings = group['user_rating'].dropna()
                if len(ratings) >= 2:
                    first_half_avg = ratings.iloc[:len(ratings)//2].mean()
                    second_half_avg = ratings.iloc[len(ratings)//2:].mean()
                    
                    if second_half_avg > first_half_avg + 0.5:
                        satisfaction_patterns['improving'] += 1
                    elif second_half_avg < first_half_avg - 0.5:
                        satisfaction_patterns['declining'] += 1
                    else:
                        satisfaction_patterns['stable'] += 1

        return satisfaction_patterns

    def _find_common_sequences(self, sessions, max_length: int = 3) -> List[Dict[str, Any]]:
        """Find common query strategy sequences."""
        sequences = defaultdict(int)
        for session_id, group in sessions:
            strategies = group['query_strategy'].tolist()
            for length in range(2, min(len(strategies) + 1, max_length + 1)):
                for i in range(len(strategies) - length + 1):
                    sequence = ' -> '.join(strategies[i:i+length])
                    sequences[sequence] += 1

        return [
            {'sequence': seq, 'count': count}
            for seq, count in sorted(sequences.items(), key=lambda x: x[1], reverse=True)[:5]
        ]

    def _count_active_sessions_today(self, df: pd.DataFrame) -> int:
        """Count sessions active today."""
        today = datetime.now().date()
        today_sessions = df[pd.to_datetime(df['timestamp']).dt.date == today]['session_id'].nunique()
        return today_sessions

    def _generate_performance_summary(self, patterns: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable performance summary."""
        total_queries = patterns.get('total_queries', 0)
        strategy_dist = patterns.get('strategy_distribution', {})
        most_used_strategy = max(strategy_dist.items(), key=lambda x: x[1])[0] if strategy_dist else 'N/A'
        
        return {
            'overview': f"Processed {total_queries} queries with {most_used_strategy} being the most used strategy",
            'efficiency': f"Average processing time varies by strategy, with room for optimization",
            'user_satisfaction': "User ratings show areas for improvement in specific strategies"
        }

    def _generate_optimization_suggestions(self, patterns: Dict[str, Any], journey: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions."""
        suggestions = []
        
        # Analyze processing times by strategy
        processing_times = patterns.get('avg_processing_time_by_strategy', {})
        if processing_times:
            slowest_strategy = max(processing_times.items(), key=lambda x: x[1])
            if slowest_strategy[1] > 3.0:  # If > 3 seconds
                suggestions.append(f"Optimize {slowest_strategy[0]} strategy - currently averaging {slowest_strategy[1]:.2f}s")

        # Analyze ratings by strategy
        ratings = patterns.get('rating_by_strategy', {})
        if ratings:
            lowest_rated = min(ratings.items(), key=lambda x: x[1])
            if lowest_rated[1] < 3.5:  # If rating < 3.5
                suggestions.append(f"Improve {lowest_rated[0]} strategy quality - current rating: {lowest_rated[1]:.2f}/5")

        # Session analysis
        avg_queries = journey.get('session_analysis', {}).get('avg_queries_per_session', 0)
        if avg_queries < 2:
            suggestions.append("Consider improving user engagement - users typically ask only one question per session")

        return suggestions[:5]  # Return top 5 suggestions

    def _detect_anomalies(self, patterns: Dict[str, Any]) -> List[str]:
        """Detect system anomalies."""
        anomalies = []
        
        # Check for unusual strategy distribution
        strategy_dist = patterns.get('strategy_distribution', {})
        if strategy_dist:
            total = sum(strategy_dist.values())
            for strategy, count in strategy_dist.items():
                percentage = (count / total) * 100
                if percentage > 80:
                    anomalies.append(f"Unusual concentration in {strategy} strategy ({percentage:.1f}%)")

        return anomalies[:3]

    def _analyze_trends(self, patterns: Dict[str, Any]) -> Dict[str, str]:
        """Analyze performance trends."""
        trends = patterns.get('performance_trends', {})
        rating_trend = trends.get('rating_trend', 'stable')
        
        return {
            'satisfaction_trend': rating_trend,
            'usage_pattern': 'Growing' if patterns.get('total_queries', 0) > 100 else 'Moderate',
            'complexity_trend': 'Increasing complexity in user queries'
        }

    def _analyze_satisfaction_insights(self, patterns: Dict[str, Any], journey: Dict[str, Any]) -> List[str]:
        """Generate satisfaction insights."""
        insights = []
        
        ratings_by_strategy = patterns.get('rating_by_strategy', {})
        if ratings_by_strategy:
            best_strategy = max(ratings_by_strategy.items(), key=lambda x: x[1])
            insights.append(f"{best_strategy[0]} strategy performs best with {best_strategy[1]:.2f}/5 rating")

        satisfaction_journey = journey.get('session_analysis', {}).get('satisfaction_journey', {})
        improving = satisfaction_journey.get('improving', 0)
        declining = satisfaction_journey.get('declining', 0)
        
        if improving > declining:
            insights.append("Users generally become more satisfied during their session")
        elif declining > improving:
            insights.append("User satisfaction tends to decline during sessions - investigate fatigue factors")

        return insights


class EnhancedFeedbackDatabase:
    """Enhanced feedback database with advanced analytics."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(FEEDBACK_DB)
        self.analytics = AdvancedAnalytics(self)
        self.init_database()

    def get_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)

    def init_database(self):
        """Initialize enhanced database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Enhanced query feedback table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS query_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT NOT NULL,
            answer_text TEXT,
            user_rating INTEGER,
            retrieval_score REAL,
            processing_time REAL,
            chunks_used INTEGER,
            chunks_data TEXT,
            feedback_text TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT,
            query_strategy TEXT,
            query_complexity_score REAL,
            user_agent TEXT,
            ip_address TEXT
        )
        ''')

        # Enhanced query cache table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS query_cache (
            query_hash TEXT PRIMARY KEY,
            query_text TEXT,
            result_data TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            access_count INTEGER DEFAULT 1,
            last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
            cache_hit_count INTEGER DEFAULT 0,
            strategy_used TEXT
        )
        ''')

        # New analytics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS analytics_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE DEFAULT (date('now')),
            total_queries INTEGER DEFAULT 0,
            avg_rating REAL DEFAULT 0.0,
            avg_processing_time REAL DEFAULT 0.0,
            strategy_distribution TEXT,
            top_keywords TEXT,
            unique_users INTEGER DEFAULT 0,
            cache_hit_rate REAL DEFAULT 0.0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # System performance table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            cpu_usage REAL,
            memory_usage REAL,
            active_sessions INTEGER,
            queries_per_minute REAL,
            error_rate REAL,
            avg_response_time REAL
        )
        ''')

        conn.commit()
        conn.close()

    def store_feedback(self, feedback_data: Dict[str, Any]):
        """Store enhanced feedback with analytics data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        def sanitize_value(value):
            if value is None:
                return None
            if isinstance(value, (int, float)) and (value != value or value == float('inf') or value == float('-inf')):
                return None
            return value

        # Sanitize chunks_data before JSON serialization
        chunks_data = sanitize_for_json(feedback_data.get('chunks_data', []))

        cursor.execute('''
        INSERT INTO query_feedback
        (query_text, answer_text, user_rating, retrieval_score,
         processing_time, chunks_used, chunks_data, feedback_text,
         session_id, query_strategy, query_complexity_score, user_agent, ip_address)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback_data.get('query'),
            feedback_data.get('answer'),
            sanitize_value(feedback_data.get('rating')),
            sanitize_value(feedback_data.get('retrieval_score')),
            sanitize_value(feedback_data.get('processing_time')),
            sanitize_value(feedback_data.get('chunks_used')),
            json.dumps(chunks_data),
            feedback_data.get('feedback_text'),
            feedback_data.get('session_id'),
            feedback_data.get('query_strategy'),
            sanitize_value(feedback_data.get('query_complexity_score')),
            feedback_data.get('user_agent'),
            feedback_data.get('ip_address')
        ))

        conn.commit()
        conn.close()

    def get_performance_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Get enhanced performance metrics."""
        conn = sqlite3.connect(self.db_path)
        query = '''
        SELECT
            AVG(CASE WHEN user_rating IS NOT NULL THEN user_rating END) as avg_rating,
            COUNT(*) as total_queries,
            AVG(CASE WHEN processing_time IS NOT NULL THEN processing_time END) as avg_response_time,
            SUM(CASE WHEN user_rating >= 4 THEN 1 ELSE 0 END) as high_rated,
            SUM(CASE WHEN user_rating <= 2 THEN 1 ELSE 0 END) as low_rated,
            COUNT(DISTINCT session_id) as unique_sessions,
            COUNT(DISTINCT query_strategy) as strategies_used
        FROM query_feedback
        WHERE timestamp >= datetime('now', '-{} days')
        '''.format(days)

        df = pd.read_sql_query(query, conn)

        # Get cache statistics
        cache_query = '''
        SELECT
            COUNT(*) as total_cached_queries,
            AVG(access_count) as avg_access_count,
            SUM(cache_hit_count) as total_cache_hits
        FROM query_cache
        WHERE timestamp >= datetime('now', '-{} days')
        '''.format(days)

        cache_df = pd.read_sql_query(cache_query, conn)
        conn.close()

        if df.empty or df.iloc[0]['total_queries'] == 0:
            return self._empty_metrics()

        metrics = df.iloc[0]
        cache_metrics = cache_df.iloc[0] if not cache_df.empty else {}

        def safe_round(value, decimals=2):
            if value is None or pd.isna(value):
                return 0.0
            try:
                return round(float(value), decimals)
            except (TypeError, ValueError):
                return 0.0

        def safe_int(value):
            if value is None or pd.isna(value):
                return 0
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        total_queries = safe_int(metrics['total_queries'])
        high_rated = safe_int(metrics['high_rated'])
        success_rate = (high_rated / total_queries * 100) if total_queries > 0 else 0.0

        # Enhanced metrics
        enhanced_metrics = {
            'avg_rating': safe_round(metrics['avg_rating']),
            'total_queries': total_queries,
            'avg_response_time': safe_round(metrics['avg_response_time']),
            'success_rate': safe_round(success_rate, 1),
            'high_rated_queries': high_rated,
            'low_rated_queries': safe_int(metrics['low_rated']),
            'unique_sessions': safe_int(metrics['unique_sessions']),
            'strategies_used': safe_int(metrics['strategies_used']),
            'queries_per_session': safe_round(total_queries / max(safe_int(metrics['unique_sessions']), 1)),
            'cache_hit_rate': safe_round(
                safe_int(cache_metrics.get('total_cache_hits', 0)) / max(total_queries, 1) * 100
            ),
            'total_cached_queries': safe_int(cache_metrics.get('total_cached_queries', 0)),
            'avg_cache_access': safe_round(cache_metrics.get('avg_access_count', 0))
        }

        return sanitize_for_json(enhanced_metrics)

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure."""
        return {
            'avg_rating': 0.0,
            'total_queries': 0,
            'avg_response_time': 0.0,
            'success_rate': 0.0,
            'high_rated_queries': 0,
            'low_rated_queries': 0,
            'unique_sessions': 0,
            'strategies_used': 0,
            'queries_per_session': 0.0,
            'cache_hit_rate': 0.0,
            'total_cached_queries': 0,
            'avg_cache_access': 0.0
        }

    def cache_query_result(self, query_hash: str, query_text: str, result: Dict[str, Any]):
        """Enhanced query result caching."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Sanitize result before JSON serialization
        sanitized_result = sanitize_for_json(result)
        strategy_used = result.get('query_strategy', 'unknown')

        cursor.execute('''
        INSERT OR REPLACE INTO query_cache
        (query_hash, query_text, result_data, access_count, last_accessed, strategy_used)
        VALUES (?, ?, ?,
                COALESCE((SELECT access_count + 1 FROM query_cache WHERE query_hash = ?), 1),
                CURRENT_TIMESTAMP, ?)
        ''', (query_hash, query_text, json.dumps(sanitized_result), query_hash, strategy_used))

        conn.commit()
        conn.close()

    def get_cached_result(self, query_hash: str) -> Optional[Dict[str, Any]]:
        """Enhanced cached result retrieval."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        SELECT result_data FROM query_cache
        WHERE query_hash = ? AND
              timestamp >= datetime('now', '-1 hour')
        ''', (query_hash,))

        result = cursor.fetchone()

        if result:
            # Update cache hit count
            cursor.execute('''
            UPDATE query_cache
            SET cache_hit_count = cache_hit_count + 1,
                last_accessed = CURRENT_TIMESTAMP
            WHERE query_hash = ?
            ''', (query_hash,))

            conn.commit()

        conn.close()

        if result:
            try:
                return json.loads(result[0])
            except json.JSONDecodeError:
                return None

        return None

    def get_optimization_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get optimization-specific analytics."""
        conn = sqlite3.connect(self.db_path)
        
        # Query for optimization data from chunks_data
        query = '''
        SELECT chunks_data, query_strategy, processing_time, user_rating
        FROM query_feedback
        WHERE timestamp >= datetime('now', '-{} days')
        AND chunks_data IS NOT NULL
        '''.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return {
                'cost_savings': {'total_saved': 0, 'percentage_saved': 0},
                'optimization_usage': {'progressive_retrieval': 0, 'aggregation_sampling': 0},
                'strategy_performance': {}
            }

        # Analyze optimization data
        total_savings = 0
        optimization_usage = {'progressive_retrieval': 0, 'aggregation_sampling': 0}
        
        for _, row in df.iterrows():
            try:
                chunks_data = json.loads(row['chunks_data']) if row['chunks_data'] else []
                
                # Look for optimization metadata in chunks
                for chunk in chunks_data:
                    if isinstance(chunk, dict):
                        if chunk.get('retrieval_method') == 'progressive_retrieval':
                            optimization_usage['progressive_retrieval'] += 1
                        if chunk.get('optimization_used'):
                            total_savings += chunk.get('chunks_saved', 0)
                            
            except (json.JSONDecodeError, TypeError):
                continue

        analytics_result = {
            'cost_savings': {
                'total_saved': total_savings,
                'percentage_saved': min(total_savings / max(len(df), 1) * 100, 100)
            },
            'optimization_usage': optimization_usage,
            'strategy_performance': df.groupby('query_strategy')['processing_time'].mean().to_dict()
        }

        return sanitize_for_json(analytics_result)

    def get_analytics_insights(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive analytics insights."""
        return self.analytics.generate_performance_insights()

    def get_query_patterns(self, days: int = 30) -> Dict[str, Any]:
        """Get detailed query pattern analysis."""
        return self.analytics.analyze_query_patterns(days)

    def get_user_journey_analysis(self) -> Dict[str, Any]:
        """Get user journey analysis."""
        return self.analytics.analyze_user_journey()

    def record_system_performance(self, performance_data: Dict[str, Any]):
        """Record system performance metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Sanitize performance data before insertion
        sanitized_data = sanitize_for_json(performance_data)

        cursor.execute('''
        INSERT INTO system_performance
        (cpu_usage, memory_usage, active_sessions, queries_per_minute, error_rate, avg_response_time)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            sanitized_data.get('cpu_usage'),
            sanitized_data.get('memory_usage'),
            sanitized_data.get('active_sessions'),
            sanitized_data.get('queries_per_minute'),
            sanitized_data.get('error_rate'),
            sanitized_data.get('avg_response_time')
        ))

        conn.commit()
        conn.close()
