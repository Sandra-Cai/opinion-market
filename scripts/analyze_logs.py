#!/usr/bin/env python3
"""
Log Analysis Script
Analyzes application logs for patterns, errors, and performance insights
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Optional advanced dependencies
try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

@dataclass
class LogEntry:
    """Represents a single log entry"""
    timestamp: datetime
    level: str
    message: str
    source: str
    raw_line: str

@dataclass
class LogAnalysisResult:
    """Result of log analysis"""
    total_entries: int
    error_count: int
    warning_count: int
    info_count: int
    debug_count: int
    unique_errors: List[str]
    error_frequency: Dict[str, int]
    performance_metrics: Dict[str, Any]
    security_events: List[str]
    recommendations: List[str]
    anomalies: Optional[Dict[str, Any]] = None
    business_metrics: Optional[Dict[str, Any]] = None
    correlation_patterns: Optional[Dict[str, Any]] = None

class LogAnalyzer:
    """Comprehensive log analyzer"""
    
    def __init__(self, log_directory: str = "logs"):
        self.log_directory = Path(log_directory)
        self.log_entries: List[LogEntry] = []
        self.analysis_result: Optional[LogAnalysisResult] = None
        
        # Common log patterns
        self.log_patterns = {
            'fastapi': r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (.+)',
            'uvicorn': r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (.+)',
            'generic': r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)',
            'json': r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z) (.+)'
        }
        
        # Error patterns
        self.error_patterns = [
            r'ERROR',
            r'CRITICAL',
            r'Exception',
            r'Traceback',
            r'Failed',
            r'Error',
            r'Timeout',
            r'Connection refused',
            r'Permission denied',
            r'Not found',
            r'Unauthorized',
            r'Forbidden'
        ]
        
        # Performance patterns
        self.performance_patterns = [
            r'(\d+\.\d+)ms',
            r'(\d+)ms',
            r'response_time[:\s]+(\d+\.?\d*)',
            r'duration[:\s]+(\d+\.?\d*)',
            r'latency[:\s]+(\d+\.?\d*)'
        ]
        
        # Security patterns
        self.security_patterns = [
            r'SQL injection',
            r'XSS',
            r'CSRF',
            r'Authentication failed',
            r'Authorization failed',
            r'Invalid token',
            r'Rate limit exceeded',
            r'Suspicious activity',
            r'Brute force',
            r'Unauthorized access'
        ]
        
        # Advanced analysis patterns
        self.anomaly_patterns = [
            r'(\d+\.\d+)ms',  # Response times
            r'(\d+) requests',  # Request counts
            r'(\d+) errors',  # Error counts
            r'(\d+)% CPU',  # CPU usage
            r'(\d+)% memory',  # Memory usage
        ]
        
        # Business logic patterns
        self.business_patterns = [
            r'trade executed',
            r'market created',
            r'user registered',
            r'payment processed',
            r'order placed',
            r'position opened',
            r'position closed'
        ]
    
    def load_logs(self, days_back: int = 7) -> int:
        """Load log files from the specified directory"""
        print(f"📁 Loading logs from {self.log_directory}...")
        
        if not self.log_directory.exists():
            print(f"❌ Log directory not found: {self.log_directory}")
            return 0
        
        log_files = list(self.log_directory.rglob("*.log"))
        log_files.extend(list(self.log_directory.rglob("*.txt")))
        
        if not log_files:
            print("❌ No log files found")
            return 0
        
        total_entries = 0
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for log_file in log_files:
            print(f"   📄 Processing: {log_file.name}")
            entries = self._parse_log_file(log_file, cutoff_date)
            self.log_entries.extend(entries)
            total_entries += len(entries)
        
        print(f"✅ Loaded {total_entries} log entries from {len(log_files)} files")
        return total_entries
    
    def _parse_log_file(self, file_path: Path, cutoff_date: datetime) -> List[LogEntry]:
        """Parse a single log file"""
        entries = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    entry = self._parse_log_line(line.strip(), str(file_path))
                    if entry and entry.timestamp >= cutoff_date:
                        entries.append(entry)
        except Exception as e:
            print(f"   ⚠️  Error reading {file_path}: {e}")
        
        return entries
    
    def _parse_log_line(self, line: str, source: str) -> Optional[LogEntry]:
        """Parse a single log line"""
        if not line:
            return None
        
        # Try different log patterns
        for pattern_name, pattern in self.log_patterns.items():
            match = re.match(pattern, line)
            if match:
                try:
                    if pattern_name == 'json':
                        timestamp_str, json_part = match.groups()
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        # Try to parse JSON part
                        try:
                            json_data = json.loads(json_part)
                            level = json_data.get('level', 'INFO')
                            message = json_data.get('message', json_part)
                        except:
                            level = 'INFO'
                            message = json_part
                    else:
                        timestamp_str, level, message = match.groups()
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                    
                    return LogEntry(
                        timestamp=timestamp,
                        level=level.upper(),
                        message=message,
                        source=source,
                        raw_line=line
                    )
                except Exception:
                    continue
        
        # If no pattern matches, try to extract timestamp and level
        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        level_match = re.search(r'\[?(\w+)\]?', line)
        
        if timestamp_match and level_match:
            try:
                timestamp = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                level = level_match.group(1).upper()
                message = line
                
                return LogEntry(
                    timestamp=timestamp,
                    level=level,
                    message=message,
                    source=source,
                    raw_line=line
                )
            except Exception:
                pass
        
        return None
    
    def analyze_logs(self) -> LogAnalysisResult:
        """Analyze loaded log entries"""
        print("🔍 Analyzing log entries...")
        
        if not self.log_entries:
            print("❌ No log entries to analyze")
            return LogAnalysisResult(0, 0, 0, 0, 0, [], {}, {}, [], [])
        
        # Count by level
        level_counts = Counter(entry.level for entry in self.log_entries)
        
        # Find unique errors
        error_entries = [entry for entry in self.log_entries if entry.level in ['ERROR', 'CRITICAL']]
        unique_errors = list(set(entry.message for entry in error_entries))
        
        # Count error frequency
        error_frequency = Counter(entry.message for entry in error_entries)
        
        # Analyze performance metrics
        performance_metrics = self._analyze_performance()
        
        # Find security events
        security_events = self._find_security_events()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(level_counts, error_frequency, security_events)
        
        # Run advanced analysis
        anomalies = self.detect_anomalies()
        business_metrics = self.analyze_business_metrics()
        correlation_patterns = self.analyze_correlation_patterns()
        
        self.analysis_result = LogAnalysisResult(
            total_entries=len(self.log_entries),
            error_count=level_counts.get('ERROR', 0) + level_counts.get('CRITICAL', 0),
            warning_count=level_counts.get('WARNING', 0),
            info_count=level_counts.get('INFO', 0),
            debug_count=level_counts.get('DEBUG', 0),
            unique_errors=unique_errors,
            error_frequency=dict(error_frequency.most_common(10)),
            performance_metrics=performance_metrics,
            security_events=security_events,
            recommendations=recommendations
        )
        
        # Store advanced analysis results
        self.analysis_result.anomalies = anomalies
        self.analysis_result.business_metrics = business_metrics
        self.analysis_result.correlation_patterns = correlation_patterns
        
        return self.analysis_result
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics from logs"""
        performance_data = {
            'response_times': [],
            'slow_requests': 0,
            'timeout_errors': 0,
            'connection_errors': 0
        }
        
        for entry in self.log_entries:
            # Extract response times
            for pattern in self.performance_patterns:
                matches = re.findall(pattern, entry.message)
                for match in matches:
                    try:
                        response_time = float(match)
                        performance_data['response_times'].append(response_time)
                        if response_time > 1000:  # > 1 second
                            performance_data['slow_requests'] += 1
                    except ValueError:
                        continue
            
            # Count timeout and connection errors
            if 'timeout' in entry.message.lower():
                performance_data['timeout_errors'] += 1
            if 'connection' in entry.message.lower() and 'error' in entry.message.lower():
                performance_data['connection_errors'] += 1
        
        # Calculate statistics
        if performance_data['response_times']:
            performance_data['avg_response_time'] = sum(performance_data['response_times']) / len(performance_data['response_times'])
            performance_data['max_response_time'] = max(performance_data['response_times'])
            performance_data['min_response_time'] = min(performance_data['response_times'])
        else:
            performance_data['avg_response_time'] = 0
            performance_data['max_response_time'] = 0
            performance_data['min_response_time'] = 0
        
        return performance_data
    
    def _find_security_events(self) -> List[str]:
        """Find security-related events in logs"""
        security_events = []
        
        for entry in self.log_entries:
            for pattern in self.security_patterns:
                if re.search(pattern, entry.message, re.IGNORECASE):
                    security_events.append(f"{entry.timestamp}: {entry.message[:100]}...")
                    break
        
        return security_events
    
    def _generate_recommendations(self, level_counts: Counter, error_frequency: Counter, security_events: List[str]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Error rate recommendations
        total_entries = sum(level_counts.values())
        error_rate = (level_counts.get('ERROR', 0) + level_counts.get('CRITICAL', 0)) / total_entries * 100
        
        if error_rate > 5:
            recommendations.append(f"High error rate detected: {error_rate:.1f}%. Review and fix critical errors.")
        elif error_rate > 1:
            recommendations.append(f"Moderate error rate: {error_rate:.1f}%. Monitor and address recurring issues.")
        
        # Top error recommendations
        if error_frequency:
            top_error = error_frequency.most_common(1)[0]
            if top_error[1] > 10:
                recommendations.append(f"Frequent error detected: '{top_error[0][:50]}...' ({top_error[1]} occurrences)")
        
        # Security recommendations
        if security_events:
            recommendations.append(f"Security events detected: {len(security_events)} events. Review security logs.")
        
        # Performance recommendations
        if self.analysis_result and self.analysis_result.performance_metrics:
            perf = self.analysis_result.performance_metrics
            if perf.get('slow_requests', 0) > 0:
                recommendations.append(f"Slow requests detected: {perf['slow_requests']} requests > 1s. Optimize performance.")
            
            if perf.get('timeout_errors', 0) > 0:
                recommendations.append(f"Timeout errors: {perf['timeout_errors']} occurrences. Check system resources.")
        
        return recommendations
    
    def detect_anomalies(self) -> Dict[str, Any]:
        """Detect anomalies in log data using machine learning"""
        if not SKLEARN_AVAILABLE:
            print("⚠️  Scikit-learn not available - skipping anomaly detection")
            return {"anomalies": [], "clusters": [], "outliers": []}
        
        print("🤖 Detecting anomalies using machine learning...")
        
        if not self.log_entries:
            return {"anomalies": [], "clusters": [], "outliers": []}
        
        # Extract features for anomaly detection
        features = []
        timestamps = []
        
        for entry in self.log_entries:
            feature_vector = []
            
            # Extract numerical features
            for pattern in self.anomaly_patterns:
                matches = re.findall(pattern, entry.message)
                if matches:
                    try:
                        feature_vector.append(float(matches[0]))
                    except ValueError:
                        feature_vector.append(0.0)
                else:
                    feature_vector.append(0.0)
            
            # Add categorical features
            feature_vector.append(1 if entry.level == 'ERROR' else 0)
            feature_vector.append(1 if entry.level == 'WARNING' else 0)
            feature_vector.append(1 if any(sec in entry.message.lower() for sec in ['error', 'failed', 'exception']) else 0)
            
            features.append(feature_vector)
            timestamps.append(entry.timestamp)
        
        if not features:
            return {"anomalies": [], "clusters": [], "outliers": []}
        
        # Convert to numpy array
        X = np.array(features)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Detect outliers using Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(X_scaled)
        
        # Cluster similar log entries
        clustering = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = clustering.fit_predict(X_scaled)
        
        # Identify anomalies
        anomalies = []
        for i, (entry, is_outlier, cluster) in enumerate(zip(self.log_entries, outlier_labels, cluster_labels)):
            if is_outlier == -1:  # Outlier
                anomalies.append({
                    "timestamp": entry.timestamp.isoformat(),
                    "level": entry.level,
                    "message": entry.message[:200] + "..." if len(entry.message) > 200 else entry.message,
                    "source": entry.source,
                    "cluster": int(cluster),
                    "anomaly_score": float(iso_forest.decision_function([X_scaled[i]])[0])
                })
        
        # Group by clusters
        clusters = defaultdict(list)
        for i, cluster in enumerate(cluster_labels):
            if cluster != -1:  # Not noise
                clusters[cluster].append({
                    "timestamp": self.log_entries[i].timestamp.isoformat(),
                    "level": self.log_entries[i].level,
                    "message": self.log_entries[i].message[:100] + "..." if len(self.log_entries[i].message) > 100 else self.log_entries[i].message
                })
        
        return {
            "anomalies": anomalies,
            "clusters": dict(clusters),
            "outliers": [a for a in anomalies if a["anomaly_score"] < -0.5]
        }
    
    def analyze_business_metrics(self) -> Dict[str, Any]:
        """Analyze business metrics from logs"""
        print("📈 Analyzing business metrics...")
        
        business_events = defaultdict(int)
        hourly_activity = defaultdict(int)
        daily_activity = defaultdict(int)
        
        for entry in self.log_entries:
            # Count business events
            for pattern in self.business_patterns:
                if re.search(pattern, entry.message, re.IGNORECASE):
                    business_events[pattern] += 1
            
            # Track hourly activity
            hour_key = entry.timestamp.strftime("%Y-%m-%d %H:00")
            hourly_activity[hour_key] += 1
            
            # Track daily activity
            day_key = entry.timestamp.strftime("%Y-%m-%d")
            daily_activity[day_key] += 1
        
        # Calculate trends
        daily_values = list(daily_activity.values())
        if len(daily_values) > 1:
            trend = "increasing" if daily_values[-1] > daily_values[0] else "decreasing"
            growth_rate = ((daily_values[-1] - daily_values[0]) / daily_values[0] * 100) if daily_values[0] > 0 else 0
        else:
            trend = "stable"
            growth_rate = 0
        
        return {
            "business_events": dict(business_events),
            "hourly_activity": dict(hourly_activity),
            "daily_activity": dict(daily_activity),
            "trend": trend,
            "growth_rate": growth_rate,
            "peak_hour": max(hourly_activity.items(), key=lambda x: x[1])[0] if hourly_activity else None,
            "busiest_day": max(daily_activity.items(), key=lambda x: x[1])[0] if daily_activity else None
        }
    
    def generate_word_cloud(self, output_dir: str = "log_analysis_charts"):
        """Generate word cloud from log messages"""
        if not WORDCLOUD_AVAILABLE:
            print("⚠️  WordCloud not available - skipping word cloud generation")
            return
        
        print("☁️  Generating word cloud...")
        
        if not self.log_entries:
            print("   ⚠️  No log entries for word cloud")
            return
        
        # Combine all log messages
        all_text = " ".join(entry.message for entry in self.log_entries)
        
        # Remove common words and clean text
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        # Clean and filter text
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        text_for_cloud = " ".join(filtered_words)
        
        if not text_for_cloud:
            print("   ⚠️  No suitable text for word cloud")
            return
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(text_for_cloud)
        
        # Save word cloud
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Common Words in Log Messages', fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path / 'word_cloud.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Word cloud saved to {output_path / 'word_cloud.png'}")
    
    def create_interactive_dashboard(self, output_dir: str = "log_analysis_charts"):
        """Create interactive dashboard using Plotly"""
        if not PLOTLY_AVAILABLE:
            print("⚠️  Plotly not available - skipping interactive dashboard")
            return
        
        print("📊 Creating interactive dashboard...")
        
        if not self.log_entries:
            print("   ⚠️  No log entries for dashboard")
            return
        
        # Prepare data
        df = pd.DataFrame([
            {
                'timestamp': entry.timestamp,
                'level': entry.level,
                'message': entry.message,
                'source': entry.source
            }
            for entry in self.log_entries
        ])
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Log Levels Over Time', 'Error Distribution', 'Hourly Activity', 'Daily Trends', 'Source Distribution', 'Message Length Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Log levels over time
        df['hour'] = df['timestamp'].dt.floor('H')
        level_counts = df.groupby(['hour', 'level']).size().unstack(fill_value=0)
        
        for level in level_counts.columns:
            fig.add_trace(
                go.Scatter(x=level_counts.index, y=level_counts[level], name=level, mode='lines+markers'),
                row=1, col=1
            )
        
        # Error distribution pie chart
        error_counts = df[df['level'].isin(['ERROR', 'CRITICAL'])]['message'].value_counts().head(10)
        fig.add_trace(
            go.Pie(labels=error_counts.index, values=error_counts.values, name="Errors"),
            row=1, col=2
        )
        
        # Hourly activity
        hourly_counts = df.groupby(df['timestamp'].dt.hour).size()
        fig.add_trace(
            go.Bar(x=hourly_counts.index, y=hourly_counts.values, name="Hourly Activity"),
            row=2, col=1
        )
        
        # Daily trends
        daily_counts = df.groupby(df['timestamp'].dt.date).size()
        fig.add_trace(
            go.Scatter(x=daily_counts.index, y=daily_counts.values, name="Daily Activity", mode='lines+markers'),
            row=2, col=2
        )
        
        # Source distribution
        source_counts = df['source'].value_counts()
        fig.add_trace(
            go.Bar(x=source_counts.index, y=source_counts.values, name="Sources"),
            row=3, col=1
        )
        
        # Message length distribution
        df['message_length'] = df['message'].str.len()
        fig.add_trace(
            go.Histogram(x=df['message_length'], name="Message Length"),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Log Analysis Interactive Dashboard",
            showlegend=True
        )
        
        # Save dashboard
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        fig.write_html(str(output_path / 'interactive_dashboard.html'))
        print(f"   ✅ Interactive dashboard saved to {output_path / 'interactive_dashboard.html'}")
    
    def analyze_correlation_patterns(self) -> Dict[str, Any]:
        """Analyze correlation patterns between different log events"""
        print("🔗 Analyzing correlation patterns...")
        
        if not self.log_entries:
            return {"correlations": [], "patterns": []}
        
        # Create time series data
        df = pd.DataFrame([
            {
                'timestamp': entry.timestamp,
                'level': entry.level,
                'is_error': 1 if entry.level in ['ERROR', 'CRITICAL'] else 0,
                'is_warning': 1 if entry.level == 'WARNING' else 0,
                'has_security': 1 if any(sec in entry.message.lower() for sec in ['auth', 'security', 'unauthorized']) else 0,
                'has_performance': 1 if any(perf in entry.message.lower() for perf in ['slow', 'timeout', 'latency']) else 0
            }
            for entry in self.log_entries
        ])
        
        # Resample to hourly data
        df.set_index('timestamp', inplace=True)
        hourly_data = df.resample('H').sum()
        
        # Calculate correlations
        correlations = hourly_data.corr()
        
        # Find interesting patterns
        patterns = []
        
        # Error-Warning correlation
        if 'is_error' in correlations.columns and 'is_warning' in correlations.columns:
            error_warning_corr = correlations.loc['is_error', 'is_warning']
            if abs(error_warning_corr) > 0.5:
                patterns.append({
                    "type": "error_warning_correlation",
                    "correlation": float(error_warning_corr),
                    "description": f"Strong correlation between errors and warnings: {error_warning_corr:.2f}"
                })
        
        # Security-Performance correlation
        if 'has_security' in correlations.columns and 'has_performance' in correlations.columns:
            sec_perf_corr = correlations.loc['has_security', 'has_performance']
            if abs(sec_perf_corr) > 0.3:
                patterns.append({
                    "type": "security_performance_correlation",
                    "correlation": float(sec_perf_corr),
                    "description": f"Correlation between security and performance issues: {sec_perf_corr:.2f}"
                })
        
        return {
            "correlations": correlations.to_dict(),
            "patterns": patterns
        }
    
    def print_report(self):
        """Print analysis report"""
        if not self.analysis_result:
            print("❌ No analysis results available")
            return
        
        result = self.analysis_result
        
        print(f"\n📊 Log Analysis Report")
        print("=" * 50)
        print(f"Total Entries: {result.total_entries}")
        print(f"Errors: {result.error_count}")
        print(f"Warnings: {result.warning_count}")
        print(f"Info: {result.info_count}")
        print(f"Debug: {result.debug_count}")
        
        if result.error_frequency:
            print(f"\n🔴 Top Errors:")
            for error, count in list(result.error_frequency.items())[:5]:
                print(f"  • {error[:80]}... ({count} times)")
        
        if result.performance_metrics:
            perf = result.performance_metrics
            print(f"\n⚡ Performance Metrics:")
            print(f"  Average Response Time: {perf.get('avg_response_time', 0):.2f}ms")
            print(f"  Max Response Time: {perf.get('max_response_time', 0):.2f}ms")
            print(f"  Slow Requests (>1s): {perf.get('slow_requests', 0)}")
            print(f"  Timeout Errors: {perf.get('timeout_errors', 0)}")
        
        if result.security_events:
            print(f"\n🔒 Security Events ({len(result.security_events)}):")
            for event in result.security_events[:3]:
                print(f"  • {event}")
        
        if result.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in result.recommendations:
                print(f"  • {rec}")
    
    def generate_charts(self, output_dir: str = "log_analysis_charts"):
        """Generate visualization charts"""
        if not self.analysis_result:
            print("❌ No analysis results available for charts")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"📊 Generating charts in {output_path}...")
        
        # Log level distribution
        result = self.analysis_result
        levels = ['ERROR', 'WARNING', 'INFO', 'DEBUG']
        counts = [result.error_count, result.warning_count, result.info_count, result.debug_count]
        
        plt.figure(figsize=(10, 6))
        plt.pie(counts, labels=levels, autopct='%1.1f%%', startangle=90)
        plt.title('Log Level Distribution')
        plt.savefig(output_path / 'log_levels.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Error frequency
        if result.error_frequency:
            errors = list(result.error_frequency.keys())[:10]
            frequencies = list(result.error_frequency.values())[:10]
            
            plt.figure(figsize=(12, 6))
            plt.barh(errors, frequencies)
            plt.title('Top 10 Errors by Frequency')
            plt.xlabel('Frequency')
            plt.tight_layout()
            plt.savefig(output_path / 'error_frequency.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Performance metrics
        if result.performance_metrics and result.performance_metrics.get('response_times'):
            response_times = result.performance_metrics['response_times']
            
            plt.figure(figsize=(10, 6))
            plt.hist(response_times, bins=50, alpha=0.7)
            plt.title('Response Time Distribution')
            plt.xlabel('Response Time (ms)')
            plt.ylabel('Frequency')
            plt.axvline(result.performance_metrics.get('avg_response_time', 0), color='red', linestyle='--', label='Average')
            plt.legend()
            plt.savefig(output_path / 'response_times.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✅ Charts saved to {output_path}")
    
    def export_report(self, output_file: str):
        """Export analysis report to JSON"""
        if not self.analysis_result:
            print("❌ No analysis results to export")
            return
        
        report_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_entries": self.analysis_result.total_entries,
                "error_count": self.analysis_result.error_count,
                "warning_count": self.analysis_result.warning_count,
                "info_count": self.analysis_result.info_count,
                "debug_count": self.analysis_result.debug_count
            },
            "error_frequency": self.analysis_result.error_frequency,
            "performance_metrics": self.analysis_result.performance_metrics,
            "security_events": self.analysis_result.security_events,
            "recommendations": self.analysis_result.recommendations
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"📄 Analysis report saved to: {output_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Log Analysis Tool")
    parser.add_argument("--log-dir", default="logs", help="Log directory path")
    parser.add_argument("--days", type=int, default=7, help="Number of days to analyze")
    parser.add_argument("--output", help="Output file for JSON report")
    parser.add_argument("--charts", help="Generate charts in specified directory")
    
    args = parser.parse_args()
    
    analyzer = LogAnalyzer(args.log_dir)
    
    # Load and analyze logs
    entries_loaded = analyzer.load_logs(args.days)
    if entries_loaded == 0:
        print("❌ No log entries found to analyze")
        sys.exit(1)
    
    result = analyzer.analyze_logs()
    analyzer.print_report()
    
    # Export report if requested
    if args.output:
        analyzer.export_report(args.output)
    
    # Generate charts if requested
    if args.charts:
        analyzer.generate_charts(args.charts)
        analyzer.generate_word_cloud(args.charts)
        analyzer.create_interactive_dashboard(args.charts)

if __name__ == "__main__":
    main()
