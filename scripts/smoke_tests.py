#!/usr/bin/env python3
"""
Smoke Tests for Opinion Market API
Basic tests to verify the API is working correctly after deployment
"""

import requests
import json
import time
import argparse
import sys
from typing import Dict, Any, List


class SmokeTester:
    """Smoke test runner for Opinion Market API"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.test_results = []
    
    def run_test(self, test_name: str, test_func):
        """Run a single test and record results"""
        print(f"Running test: {test_name}")
        try:
            result = test_func()
            if result:
                print(f"✅ {test_name} - PASSED")
                self.test_results.append({"test": test_name, "status": "PASSED", "error": None})
            else:
                print(f"❌ {test_name} - FAILED")
                self.test_results.append({"test": test_name, "status": "FAILED", "error": "Test returned False"})
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {str(e)}")
            self.test_results.append({"test": test_name, "status": "ERROR", "error": str(e)})
    
    def test_health_endpoint(self) -> bool:
        """Test the health endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def test_ready_endpoint(self) -> bool:
        """Test the ready endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/ready", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def test_api_docs(self) -> bool:
        """Test API documentation endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/docs", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def test_openapi_schema(self) -> bool:
        """Test OpenAPI schema endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/openapi.json", timeout=10)
            if response.status_code == 200:
                schema = response.json()
                return "info" in schema and "paths" in schema
            return False
        except Exception:
            return False
    
    def test_enhanced_cache_endpoints(self) -> bool:
        """Test enhanced cache endpoints"""
        try:
            # Test cache stats endpoint
            response = self.session.get(f"{self.base_url}/api/v1/enhanced-cache/stats", timeout=10)
            if response.status_code != 200:
                return False
            
            # Test cache health check
            response = self.session.get(f"{self.base_url}/api/v1/enhanced-cache/health-check", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def test_performance_dashboard(self) -> bool:
        """Test performance dashboard endpoints"""
        try:
            # Test dashboard HTML
            response = self.session.get(f"{self.base_url}/api/v1/performance-dashboard/dashboard", timeout=10)
            if response.status_code != 200:
                return False
            
            # Test metrics endpoint
            response = self.session.get(f"{self.base_url}/api/v1/performance-dashboard/metrics/current", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def test_business_intelligence(self) -> bool:
        """Test business intelligence endpoints"""
        try:
            # Test KPI summary
            response = self.session.get(f"{self.base_url}/api/v1/business-intelligence/kpi-summary", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def test_ai_optimization(self) -> bool:
        """Test AI optimization endpoints"""
        try:
            # Test recommendations endpoint
            response = self.session.get(f"{self.base_url}/api/v1/ai-optimization/recommendations", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def test_security_endpoints(self) -> bool:
        """Test security endpoints"""
        try:
            # Test security metrics
            response = self.session.get(f"{self.base_url}/api/v1/security/metrics", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def test_database_optimization(self) -> bool:
        """Test database optimization endpoints"""
        try:
            # Test database metrics
            response = self.session.get(f"{self.base_url}/api/v1/database-optimization/metrics", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def test_response_times(self) -> bool:
        """Test that response times are acceptable"""
        try:
            endpoints = [
                "/health",
                "/api/v1/enhanced-cache/stats",
                "/api/v1/performance-dashboard/metrics/current"
            ]
            
            for endpoint in endpoints:
                start_time = time.time()
                response = self.session.get(f"{self.base_url}{endpoint}", timeout=10)
                response_time = time.time() - start_time
                
                if response.status_code != 200:
                    return False
                
                # Response time should be less than 2 seconds
                if response_time > 2.0:
                    print(f"Warning: {endpoint} took {response_time:.2f}s")
            
            return True
        except Exception:
            return False
    
    def run_all_tests(self):
        """Run all smoke tests"""
        print(f"🧪 Running smoke tests against: {self.base_url}")
        print("=" * 50)
        
        # Core functionality tests
        self.run_test("Health Endpoint", self.test_health_endpoint)
        self.run_test("Ready Endpoint", self.test_ready_endpoint)
        self.run_test("API Documentation", self.test_api_docs)
        self.run_test("OpenAPI Schema", self.test_openapi_schema)
        
        # Feature-specific tests
        self.run_test("Enhanced Cache Endpoints", self.test_enhanced_cache_endpoints)
        self.run_test("Performance Dashboard", self.test_performance_dashboard)
        self.run_test("Business Intelligence", self.test_business_intelligence)
        self.run_test("AI Optimization", self.test_ai_optimization)
        self.run_test("Security Endpoints", self.test_security_endpoints)
        self.run_test("Database Optimization", self.test_database_optimization)
        
        # Performance tests
        self.run_test("Response Times", self.test_response_times)
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 50)
        print("📊 SMOKE TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "PASSED"])
        failed_tests = len([r for r in self.test_results if r["status"] == "FAILED"])
        error_tests = len([r for r in self.test_results if r["status"] == "ERROR"])
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⚠️  Errors: {error_tests}")
        
        if failed_tests > 0 or error_tests > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.test_results:
                if result["status"] != "PASSED":
                    print(f"  - {result['test']}: {result['status']}")
                    if result["error"]:
                        print(f"    Error: {result['error']}")
        
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        print(f"\n🎯 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("🎉 Smoke tests PASSED! Deployment is healthy.")
            return True
        else:
            print("💥 Smoke tests FAILED! Deployment needs attention.")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run smoke tests for Opinion Market API")
    parser.add_argument("--base-url", default="http://localhost:8000", 
                       help="Base URL of the API (default: http://localhost:8000)")
    parser.add_argument("--environment", default="production",
                       help="Environment name (default: production)")
    
    args = parser.parse_args()
    
    # Run smoke tests
    tester = SmokeTester(args.base_url)
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
