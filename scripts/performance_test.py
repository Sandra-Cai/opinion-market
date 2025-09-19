#!/usr/bin/env python3
"""
Performance Testing Script
Tests API performance under various loads and scenarios
"""

import asyncio
import aiohttp
import time
import statistics
import json
from typing import List, Dict, Any
import argparse
from datetime import datetime

class PerformanceTester:
    """Performance testing class for the Opinion Market API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        self.auth_token = None
    
    async def authenticate(self, session: aiohttp.ClientSession) -> bool:
        """Authenticate and get access token"""
        try:
            # Register a test user
            user_data = {
                "username": f"perf_test_{int(time.time())}",
                "email": f"perf_test_{int(time.time())}@example.com",
                "password": "PerformanceTest123!",
                "full_name": "Performance Test User"
            }
            
            async with session.post(f"{self.base_url}/api/v1/auth/register", json=user_data) as response:
                if response.status == 201:
                    data = await response.json()
                    self.auth_token = data.get("access_token")
                    return True
                elif response.status == 409:  # User already exists
                    # Try to login instead
                    login_data = {
                        "username": user_data["username"],
                        "password": user_data["password"]
                    }
                    async with session.post(f"{self.base_url}/api/v1/auth/login", json=login_data) as login_response:
                        if login_response.status == 200:
                            data = await login_response.json()
                            self.auth_token = data.get("access_token")
                            return True
            
            return False
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False
    
    async def make_request(self, session: aiohttp.ClientSession, method: str, endpoint: str, 
                          data: Dict = None, headers: Dict = None) -> Dict[str, Any]:
        """Make a single API request and measure performance"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                async with session.get(url, headers=headers) as response:
                    response_data = await response.text()
                    status_code = response.status
            elif method.upper() == "POST":
                async with session.post(url, json=data, headers=headers) as response:
                    response_data = await response.text()
                    status_code = response.status
            elif method.upper() == "PUT":
                async with session.put(url, json=data, headers=headers) as response:
                    response_data = await response.text()
                    status_code = response.status
            elif method.upper() == "DELETE":
                async with session.delete(url, headers=headers) as response:
                    response_data = await response.text()
                    status_code = response.status
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            return {
                "method": method,
                "endpoint": endpoint,
                "status_code": status_code,
                "response_time_ms": response_time,
                "success": 200 <= status_code < 300,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            
            return {
                "method": method,
                "endpoint": endpoint,
                "status_code": 0,
                "response_time_ms": response_time,
                "success": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def test_endpoint_performance(self, session: aiohttp.ClientSession, 
                                      endpoint: str, method: str = "GET", 
                                      data: Dict = None, headers: Dict = None,
                                      iterations: int = 10) -> Dict[str, Any]:
        """Test performance of a specific endpoint"""
        print(f"🧪 Testing {method} {endpoint} ({iterations} iterations)...")
        
        results = []
        for i in range(iterations):
            result = await self.make_request(session, method, endpoint, data, headers)
            results.append(result)
            
            # Small delay between requests
            await asyncio.sleep(0.1)
        
        # Calculate statistics
        response_times = [r["response_time_ms"] for r in results]
        success_count = sum(1 for r in results if r["success"])
        
        stats = {
            "endpoint": endpoint,
            "method": method,
            "iterations": iterations,
            "success_rate": (success_count / iterations) * 100,
            "avg_response_time_ms": statistics.mean(response_times),
            "min_response_time_ms": min(response_times),
            "max_response_time_ms": max(response_times),
            "median_response_time_ms": statistics.median(response_times),
            "std_deviation_ms": statistics.stdev(response_times) if len(response_times) > 1 else 0,
            "results": results
        }
        
        self.results.append(stats)
        return stats
    
    async def test_concurrent_requests(self, session: aiohttp.ClientSession, 
                                     endpoint: str, method: str = "GET",
                                     data: Dict = None, headers: Dict = None,
                                     concurrent_requests: int = 10) -> Dict[str, Any]:
        """Test performance under concurrent load"""
        print(f"🚀 Testing {method} {endpoint} with {concurrent_requests} concurrent requests...")
        
        # Create concurrent requests
        tasks = []
        for _ in range(concurrent_requests):
            task = self.make_request(session, method, endpoint, data, headers)
            tasks.append(task)
        
        # Execute all requests concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # Process results
        valid_results = []
        for result in results:
            if isinstance(result, dict):
                valid_results.append(result)
            else:
                valid_results.append({
                    "method": method,
                    "endpoint": endpoint,
                    "status_code": 0,
                    "response_time_ms": 0,
                    "success": False,
                    "error": str(result),
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        # Calculate statistics
        response_times = [r["response_time_ms"] for r in valid_results]
        success_count = sum(1 for r in valid_results if r["success"])
        total_time = (end_time - start_time) * 1000
        
        stats = {
            "endpoint": endpoint,
            "method": method,
            "concurrent_requests": concurrent_requests,
            "total_time_ms": total_time,
            "success_rate": (success_count / concurrent_requests) * 100,
            "avg_response_time_ms": statistics.mean(response_times),
            "min_response_time_ms": min(response_times),
            "max_response_time_ms": max(response_times),
            "median_response_time_ms": statistics.median(response_times),
            "std_deviation_ms": statistics.stdev(response_times) if len(response_times) > 1 else 0,
            "requests_per_second": (concurrent_requests / total_time) * 1000,
            "results": valid_results
        }
        
        self.results.append(stats)
        return stats
    
    async def run_comprehensive_test(self):
        """Run comprehensive performance tests"""
        print("🚀 Starting Comprehensive Performance Test")
        print("=" * 60)
        
        async with aiohttp.ClientSession() as session:
            # Authenticate
            print("🔐 Authenticating...")
            if not await self.authenticate(session):
                print("❌ Authentication failed. Exiting.")
                return
            
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            print("✅ Authentication successful")
            
            # Test basic endpoints
            print("\n📊 Testing Basic Endpoints...")
            await self.test_endpoint_performance(session, "/api/v1/health", "GET")
            await self.test_endpoint_performance(session, "/api/v1/health/detailed", "GET")
            await self.test_endpoint_performance(session, "/api/v1/markets/", "GET")
            await self.test_endpoint_performance(session, "/api/v1/auth/me", "GET", headers=headers)
            
            # Test documentation endpoints
            print("\n📚 Testing Documentation Endpoints...")
            await self.test_endpoint_performance(session, "/api/v1/docs/overview", "GET")
            await self.test_endpoint_performance(session, "/api/v1/docs/examples", "GET")
            await self.test_endpoint_performance(session, "/api/v1/docs/status-codes", "GET")
            
            # Test market creation
            print("\n🏪 Testing Market Operations...")
            market_data = {
                "title": f"Performance Test Market {int(time.time())}",
                "description": "A market created for performance testing",
                "outcome_a": "Yes",
                "outcome_b": "No",
                "end_date": "2024-12-31T23:59:59Z",
                "category": "test",
                "tags": ["performance", "test"]
            }
            await self.test_endpoint_performance(session, "/api/v1/markets/", "POST", 
                                               market_data, headers, iterations=5)
            
            # Test concurrent load
            print("\n⚡ Testing Concurrent Load...")
            await self.test_concurrent_requests(session, "/api/v1/health", "GET", 
                                              concurrent_requests=20)
            await self.test_concurrent_requests(session, "/api/v1/markets/", "GET", 
                                              concurrent_requests=15)
            await self.test_concurrent_requests(session, "/api/v1/auth/me", "GET", 
                                              headers=headers, concurrent_requests=10)
            
            # Test high load
            print("\n🔥 Testing High Load...")
            await self.test_concurrent_requests(session, "/api/v1/health", "GET", 
                                              concurrent_requests=50)
            await self.test_concurrent_requests(session, "/api/v1/docs/overview", "GET", 
                                              concurrent_requests=30)
    
    def generate_report(self):
        """Generate performance test report"""
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE TEST REPORT")
        print("=" * 60)
        
        if not self.results:
            print("❌ No test results available")
            return
        
        # Overall statistics
        all_response_times = []
        all_success_rates = []
        
        for result in self.results:
            if "avg_response_time_ms" in result:
                all_response_times.append(result["avg_response_time_ms"])
            if "success_rate" in result:
                all_success_rates.append(result["success_rate"])
        
        print(f"📈 Overall Statistics:")
        print(f"   Total Tests: {len(self.results)}")
        print(f"   Average Response Time: {statistics.mean(all_response_times):.2f} ms")
        print(f"   Average Success Rate: {statistics.mean(all_success_rates):.2f}%")
        print(f"   Min Response Time: {min(all_response_times):.2f} ms")
        print(f"   Max Response Time: {max(all_response_times):.2f} ms")
        
        # Detailed results
        print(f"\n📋 Detailed Results:")
        for i, result in enumerate(self.results, 1):
            print(f"\n{i}. {result['method']} {result['endpoint']}")
            print(f"   Success Rate: {result.get('success_rate', 'N/A'):.2f}%")
            print(f"   Avg Response Time: {result.get('avg_response_time_ms', 'N/A'):.2f} ms")
            print(f"   Min/Max: {result.get('min_response_time_ms', 'N/A'):.2f} / {result.get('max_response_time_ms', 'N/A'):.2f} ms")
            
            if "requests_per_second" in result:
                print(f"   Requests/sec: {result['requests_per_second']:.2f}")
            
            if "concurrent_requests" in result:
                print(f"   Concurrent Requests: {result['concurrent_requests']}")
        
        # Performance recommendations
        print(f"\n💡 Performance Recommendations:")
        avg_response_time = statistics.mean(all_response_times)
        avg_success_rate = statistics.mean(all_success_rates)
        
        if avg_response_time > 1000:
            print("   ⚠️  Average response time is high (>1000ms). Consider optimization.")
        elif avg_response_time > 500:
            print("   ⚠️  Average response time is moderate (>500ms). Monitor performance.")
        else:
            print("   ✅ Average response time is good (<500ms).")
        
        if avg_success_rate < 95:
            print("   ⚠️  Success rate is low (<95%). Check for errors.")
        else:
            print("   ✅ Success rate is good (>95%).")
        
        # Save results to file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_test_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": datetime.utcnow().isoformat(),
                "summary": {
                    "total_tests": len(self.results),
                    "avg_response_time_ms": statistics.mean(all_response_times),
                    "avg_success_rate": statistics.mean(all_success_rates),
                    "min_response_time_ms": min(all_response_times),
                    "max_response_time_ms": max(all_response_times)
                },
                "results": self.results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Performance Test for Opinion Market API")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL of the API (default: http://localhost:8000)")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick test with fewer iterations")
    
    args = parser.parse_args()
    
    tester = PerformanceTester(args.url)
    
    try:
        await tester.run_comprehensive_test()
        tester.generate_report()
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
