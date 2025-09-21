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
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import seaborn as sns
from wordcloud import WordCloud
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

if __name__ == "__main__":
    main()
