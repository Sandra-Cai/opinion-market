#!/usr/bin/env python3
"""
Performance Benchmarking Script
Comprehensive performance testing for the Opinion Market API
"""

import asyncio
import aiohttp
import time
import json
import statistics
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class BenchmarkResult:
    """Result of a single benchmark test"""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    requests_per_second: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    error_rate: float
    timestamp: str

@dataclass
class RequestResult:
    """Result of a single HTTP request"""
    status_code: int
    response_time: float
    success: bool
    error: Optional[str] = None

class PerformanceBenchmark:
    """Performance benchmarking tool"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        self.auth_token: Optional[str] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def authenticate(self, username: str = "testuser", password: str = "TestPassword123!"):
        """Authenticate and get access token"""
        try:
            # Try to register first
            register_data = {
                "username": username,
                "email": f"{username}@example.com",
                "password": password,
                "full_name": "Test User"
            }
            
            async with self.session.post(f"{self.base_url}/api/v1/auth/register", json=register_data) as response:
                if response.status == 201:
                    data = await response.json()
                    self.auth_token = data.get("access_token")
                    print(f"✅ User registered and authenticated: {username}")
                elif response.status == 400:
                    # User might already exist, try to login
                    login_data = {
                        "username": username,
                        "password": password
                    }
                    async with self.session.post(f"{self.base_url}/api/v1/auth/login", json=login_data) as login_response:
                        if login_response.status == 200:
                            data = await login_response.json()
                            self.auth_token = data.get("access_token")
                            print(f"✅ User logged in: {username}")
                        else:
                            raise Exception(f"Login failed: {login_response.status}")
                else:
                    raise Exception(f"Registration failed: {response.status}")
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            raise
    
    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers with authentication"""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers
    
    async def make_request(self, method: str, endpoint: str, **kwargs) -> RequestResult:
        """Make a single HTTP request"""
        start_time = time.perf_counter()
        
        try:
            url = f"{self.base_url}{endpoint}"
            headers = self.get_headers()
            
            async with self.session.request(method, url, headers=headers, **kwargs) as response:
                end_time = time.perf_counter()
                response_time = (end_time - start_time) * 1000  # Convert to milliseconds
                
                return RequestResult(
                    status_code=response.status,
                    response_time=response_time,
                    success=200 <= response.status < 400
                )
        except Exception as e:
            end_time = time.perf_counter()
            response_time = (end_time - start_time) * 1000
            
            return RequestResult(
                status_code=0,
                response_time=response_time,
                success=False,
                error=str(e)
            )
    
    async def run_load_test(self, test_name: str, requests: List[Dict[str, Any]], 
                          concurrent_users: int = 10, duration_seconds: int = 60) -> BenchmarkResult:
        """Run a load test with specified parameters"""
        print(f"\n🧪 Running load test: {test_name}")
        print(f"   Concurrent users: {concurrent_users}")
        print(f"   Duration: {duration_seconds} seconds")
        print(f"   Total requests: {len(requests)}")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_users)
        
        async def make_request_with_semaphore(request_data):
            async with semaphore:
                return await self.make_request(**request_data)
        
        # Run requests
        tasks = []
        request_index = 0
        
        while time.time() < end_time:
            if request_index < len(requests):
                request_data = requests[request_index]
                task = asyncio.create_task(make_request_with_semaphore(request_data))
                tasks.append(task)
                request_index = (request_index + 1) % len(requests)
            else:
                # If we've used all requests, wait a bit before restarting
                await asyncio.sleep(0.1)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and process results
        valid_results = []
        for result in results:
            if isinstance(result, RequestResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                valid_results.append(RequestResult(
                    status_code=0,
                    response_time=0,
                    success=False,
                    error=str(result)
                ))
        
        # Calculate statistics
        total_time = time.time() - start_time
        successful_requests = sum(1 for r in valid_results if r.success)
        failed_requests = len(valid_results) - successful_requests
        
        response_times = [r.response_time for r in valid_results if r.response_time > 0]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p50_response_time = statistics.median(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p50_response_time = p95_response_time = p99_response_time = 0
        
        requests_per_second = len(valid_results) / total_time if total_time > 0 else 0
        error_rate = (failed_requests / len(valid_results) * 100) if valid_results else 0
        
        result = BenchmarkResult(
            test_name=test_name,
            total_requests=len(valid_results),
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            requests_per_second=requests_per_second,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            error_rate=error_rate,
            timestamp=datetime.utcnow().isoformat()
        )
        
        self.print_benchmark_result(result)
        return result
    
    def print_benchmark_result(self, result: BenchmarkResult):
        """Print benchmark result in a formatted way"""
        print(f"\n📊 Benchmark Results: {result.test_name}")
        print("=" * 50)
        print(f"Total Requests: {result.total_requests}")
        print(f"Successful: {result.successful_requests}")
        print(f"Failed: {result.failed_requests}")
        print(f"Error Rate: {result.error_rate:.2f}%")
        print(f"Requests/sec: {result.requests_per_second:.2f}")
        print(f"Total Time: {result.total_time:.2f}s")
        print(f"\nResponse Times (ms):")
        print(f"  Average: {result.avg_response_time:.2f}")
        print(f"  Min: {result.min_response_time:.2f}")
        print(f"  Max: {result.max_response_time:.2f}")
        print(f"  P50: {result.p50_response_time:.2f}")
        print(f"  P95: {result.p95_response_time:.2f}")
        print(f"  P99: {result.p99_response_time:.2f}")
    
    async def test_health_endpoints(self, concurrent_users: int = 10, duration: int = 30) -> BenchmarkResult:
        """Test health check endpoints"""
        requests = [
            {"method": "GET", "endpoint": "/api/v1/health"},
            {"method": "GET", "endpoint": "/api/v1/health/detailed"},
            {"method": "GET", "endpoint": "/api/v1/health/readiness"},
            {"method": "GET", "endpoint": "/api/v1/health/liveness"},
        ]
        
        return await self.run_load_test(
            "Health Endpoints",
            requests,
            concurrent_users=concurrent_users,
            duration_seconds=duration
        )
    
    async def test_market_endpoints(self, concurrent_users: int = 10, duration: int = 60) -> BenchmarkResult:
        """Test market-related endpoints"""
        requests = [
            {"method": "GET", "endpoint": "/api/v1/markets/"},
            {"method": "GET", "endpoint": "/api/v1/markets/?limit=10"},
            {"method": "GET", "endpoint": "/api/v1/markets/?category=test"},
            {"method": "GET", "endpoint": "/api/v1/analytics/markets"},
        ]
        
        return await self.run_load_test(
            "Market Endpoints",
            requests,
            concurrent_users=concurrent_users,
            duration_seconds=duration
        )
    
    async def test_user_endpoints(self, concurrent_users: int = 10, duration: int = 60) -> BenchmarkResult:
        """Test user-related endpoints"""
        requests = [
            {"method": "GET", "endpoint": "/api/v1/auth/me"},
            {"method": "GET", "endpoint": "/api/v1/users/"},
            {"method": "GET", "endpoint": "/api/v1/leaderboard/"},
        ]
        
        return await self.run_load_test(
            "User Endpoints",
            requests,
            concurrent_users=concurrent_users,
            duration_seconds=duration
        )
    
    async def test_mixed_workload(self, concurrent_users: int = 20, duration: int = 120) -> BenchmarkResult:
        """Test mixed workload with various endpoints"""
        requests = [
            {"method": "GET", "endpoint": "/api/v1/health"},
            {"method": "GET", "endpoint": "/api/v1/markets/"},
            {"method": "GET", "endpoint": "/api/v1/auth/me"},
            {"method": "GET", "endpoint": "/api/v1/analytics/markets"},
            {"method": "GET", "endpoint": "/api/v1/leaderboard/"},
            {"method": "GET", "endpoint": "/api/v1/health/detailed"},
        ]
        
        return await self.run_load_test(
            "Mixed Workload",
            requests,
            concurrent_users=concurrent_users,
            duration_seconds=duration
        )
    
    async def test_high_load(self, concurrent_users: int = 50, duration: int = 60) -> BenchmarkResult:
        """Test high load scenario"""
        requests = [
            {"method": "GET", "endpoint": "/api/v1/health"},
            {"method": "GET", "endpoint": "/api/v1/markets/"},
        ]
        
        return await self.run_load_test(
            "High Load",
            requests,
            concurrent_users=concurrent_users,
            duration_seconds=duration
        )
    
    def generate_report(self, results: List[BenchmarkResult], output_file: str):
        """Generate a comprehensive performance report"""
        report_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_tests": len(results),
                "total_requests": sum(r.total_requests for r in results),
                "total_successful": sum(r.successful_requests for r in results),
                "total_failed": sum(r.failed_requests for r in results),
                "avg_requests_per_second": statistics.mean([r.requests_per_second for r in results]),
                "avg_response_time": statistics.mean([r.avg_response_time for r in results]),
                "max_error_rate": max([r.error_rate for r in results])
            },
            "tests": [asdict(result) for result in results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Performance report saved to: {output_file}")
    
    def plot_results(self, results: List[BenchmarkResult], output_file: str):
        """Generate performance charts"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        test_names = [r.test_name for r in results]
        
        # Requests per second
        ax1.bar(test_names, [r.requests_per_second for r in results])
        ax1.set_title('Requests per Second')
        ax1.set_ylabel('RPS')
        ax1.tick_params(axis='x', rotation=45)
        
        # Average response time
        ax2.bar(test_names, [r.avg_response_time for r in results])
        ax2.set_title('Average Response Time')
        ax2.set_ylabel('ms')
        ax2.tick_params(axis='x', rotation=45)
        
        # Error rate
        ax3.bar(test_names, [r.error_rate for r in results])
        ax3.set_title('Error Rate')
        ax3.set_ylabel('%')
        ax3.tick_params(axis='x', rotation=45)
        
        # P95 response time
        ax4.bar(test_names, [r.p95_response_time for r in results])
        ax4.set_title('P95 Response Time')
        ax4.set_ylabel('ms')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Performance charts saved to: {output_file}")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Performance Benchmarking Tool")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--username", default="testuser", help="Username for authentication")
    parser.add_argument("--password", default="TestPassword123!", help="Password for authentication")
    parser.add_argument("--concurrent", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file for results")
    parser.add_argument("--charts", default="benchmark_charts.png", help="Output file for charts")
    parser.add_argument("--tests", nargs="+", choices=["health", "markets", "users", "mixed", "high", "all"],
                       default=["all"], help="Tests to run")
    
    args = parser.parse_args()
    
    print("🚀 Performance Benchmarking Tool")
    print("=" * 50)
    print(f"Target URL: {args.url}")
    print(f"Concurrent Users: {args.concurrent}")
    print(f"Test Duration: {args.duration}s")
    print(f"Tests: {', '.join(args.tests)}")
    
    async with PerformanceBenchmark(args.url) as benchmark:
        # Authenticate
        await benchmark.authenticate(args.username, args.password)
        
        results = []
        
        # Run selected tests
        if "health" in args.tests or "all" in args.tests:
            result = await benchmark.test_health_endpoints(args.concurrent, args.duration)
            results.append(result)
        
        if "markets" in args.tests or "all" in args.tests:
            result = await benchmark.test_market_endpoints(args.concurrent, args.duration)
            results.append(result)
        
        if "users" in args.tests or "all" in args.tests:
            result = await benchmark.test_user_endpoints(args.concurrent, args.duration)
            results.append(result)
        
        if "mixed" in args.tests or "all" in args.tests:
            result = await benchmark.test_mixed_workload(args.concurrent, args.duration)
            results.append(result)
        
        if "high" in args.tests or "all" in args.tests:
            result = await benchmark.test_high_load(args.concurrent * 2, args.duration)
            results.append(result)
        
        # Generate reports
        benchmark.generate_report(results, args.output)
        benchmark.plot_results(results, args.charts)
        
        print(f"\n🎉 Benchmarking completed!")
        print(f"   Total tests: {len(results)}")
        print(f"   Total requests: {sum(r.total_requests for r in results)}")
        print(f"   Average RPS: {statistics.mean([r.requests_per_second for r in results]):.2f}")

if __name__ == "__main__":
    asyncio.run(main())
