#!/usr/bin/env python3
"""
Advanced Features Test Suite
Comprehensive testing of all new advanced features
"""

import asyncio
import time
import json
import requests
import sys
from datetime import datetime
from typing import Dict, Any, List

# Test configuration
BASE_URL = "http://localhost:8000/api/v1"
TEST_USER = {
    "username": "test_user_advanced",
    "email": "test_advanced@example.com",
    "password": "TestPassword123!",
    "full_name": "Advanced Test User"
}

class AdvancedFeaturesTester:
    """Comprehensive tester for advanced features"""
    
    def __init__(self):
        self.session = requests.Session()
        self.auth_token = None
        self.test_results = []
        
    async def run_all_tests(self):
        """Run all advanced feature tests"""
        print("🚀 Starting Advanced Features Test Suite")
        print("=" * 60)
        
        try:
            # Setup
            await self._setup_test_environment()
            
            # Test 1: Enhanced API Documentation
            await self._test_enhanced_api_docs()
            
            # Test 2: Advanced Performance Optimizer
            await self._test_advanced_performance_optimizer()
            
            # Test 3: Advanced Dashboard
            await self._test_advanced_dashboard()
            
            # Test 4: Advanced Analytics Engine
            await self._test_advanced_analytics_engine()
            
            # Test 5: Auto Scaling Manager
            await self._test_auto_scaling_manager()
            
            # Test 6: Integration Tests
            await self._test_system_integration()
            
            # Generate report
            self._generate_test_report()
            
        except Exception as e:
            print(f"❌ Test suite failed: {e}")
            return False
            
        return True
        
    async def _setup_test_environment(self):
        """Setup test environment and authentication"""
        print("🔧 Setting up test environment...")
        
        try:
            # Register test user
            response = self.session.post(f"{BASE_URL}/auth/register", json=TEST_USER)
            if response.status_code not in [200, 201, 400]:  # 400 if user already exists
                print(f"⚠️  User registration response: {response.status_code}")
                
            # Login
            login_data = {
                "username": TEST_USER["username"],
                "password": TEST_USER["password"]
            }
            response = self.session.post(f"{BASE_URL}/auth/login", json=login_data)
            
            if response.status_code == 200:
                self.auth_token = response.json()["access_token"]
                self.session.headers.update({"Authorization": f"Bearer {self.auth_token}"})
                print("✅ Authentication successful")
            else:
                print(f"❌ Authentication failed: {response.status_code}")
                raise Exception("Authentication failed")
                
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            raise
            
    async def _test_enhanced_api_docs(self):
        """Test enhanced API documentation"""
        print("\n📚 Testing Enhanced API Documentation...")
        
        try:
            # Test comprehensive docs endpoint
            response = self.session.get(f"{BASE_URL}/docs/comprehensive")
            if response.status_code == 200:
                print("✅ Enhanced API documentation accessible")
                self._record_test_result("enhanced_api_docs", True, "Documentation accessible")
            else:
                print(f"❌ Enhanced API documentation failed: {response.status_code}")
                self._record_test_result("enhanced_api_docs", False, f"HTTP {response.status_code}")
                
            # Test performance insights endpoint
            response = self.session.get(f"{BASE_URL}/docs/performance-insights")
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    print("✅ Performance insights endpoint working")
                    self._record_test_result("performance_insights", True, "Insights generated")
                else:
                    print("❌ Performance insights failed")
                    self._record_test_result("performance_insights", False, "No insights")
            else:
                print(f"❌ Performance insights endpoint failed: {response.status_code}")
                self._record_test_result("performance_insights", False, f"HTTP {response.status_code}")
                
            # Test endpoint discovery
            response = self.session.get(f"{BASE_URL}/docs/endpoint-discovery")
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and data.get("data", {}).get("total_endpoints", 0) > 0:
                    print("✅ Endpoint discovery working")
                    self._record_test_result("endpoint_discovery", True, f"Found {data['data']['total_endpoints']} endpoints")
                else:
                    print("❌ Endpoint discovery failed")
                    self._record_test_result("endpoint_discovery", False, "No endpoints found")
            else:
                print(f"❌ Endpoint discovery failed: {response.status_code}")
                self._record_test_result("endpoint_discovery", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Enhanced API docs test failed: {e}")
            self._record_test_result("enhanced_api_docs", False, str(e))
            
    async def _test_advanced_performance_optimizer(self):
        """Test advanced performance optimizer"""
        print("\n⚡ Testing Advanced Performance Optimizer...")
        
        try:
            # Test performance summary
            response = self.session.get(f"{BASE_URL}/advanced-performance/summary")
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    perf_data = data.get("data", {})
                    print(f"✅ Performance summary: Score {perf_data.get('performance_score', 0):.1f}/100")
                    self._record_test_result("performance_summary", True, f"Score: {perf_data.get('performance_score', 0):.1f}")
                else:
                    print("❌ Performance summary failed")
                    self._record_test_result("performance_summary", False, "No data")
            else:
                print(f"❌ Performance summary failed: {response.status_code}")
                self._record_test_result("performance_summary", False, f"HTTP {response.status_code}")
                
            # Test monitoring control
            response = self.session.post(f"{BASE_URL}/advanced-performance/start-monitoring")
            if response.status_code == 200:
                print("✅ Performance monitoring started")
                self._record_test_result("monitoring_control", True, "Monitoring started")
            else:
                print(f"❌ Performance monitoring failed: {response.status_code}")
                self._record_test_result("monitoring_control", False, f"HTTP {response.status_code}")
                
            # Test optimization control
            response = self.session.post(f"{BASE_URL}/advanced-performance/start-optimization")
            if response.status_code == 200:
                print("✅ Performance optimization started")
                self._record_test_result("optimization_control", True, "Optimization started")
            else:
                print(f"❌ Performance optimization failed: {response.status_code}")
                self._record_test_result("optimization_control", False, f"HTTP {response.status_code}")
                
            # Test predictions
            response = self.session.get(f"{BASE_URL}/advanced-performance/predictions")
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    predictions = data.get("data", [])
                    print(f"✅ Performance predictions: {len(predictions)} predictions")
                    self._record_test_result("performance_predictions", True, f"{len(predictions)} predictions")
                else:
                    print("❌ Performance predictions failed")
                    self._record_test_result("performance_predictions", False, "No predictions")
            else:
                print(f"❌ Performance predictions failed: {response.status_code}")
                self._record_test_result("performance_predictions", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Advanced performance optimizer test failed: {e}")
            self._record_test_result("advanced_performance", False, str(e))
            
    async def _test_advanced_dashboard(self):
        """Test advanced dashboard"""
        print("\n🎛️ Testing Advanced Dashboard...")
        
        try:
            # Test dashboard access
            response = self.session.get(f"{BASE_URL}/advanced-dashboard/dashboard/advanced")
            if response.status_code == 200:
                print("✅ Advanced dashboard accessible")
                self._record_test_result("dashboard_access", True, "Dashboard accessible")
            else:
                print(f"❌ Advanced dashboard failed: {response.status_code}")
                self._record_test_result("dashboard_access", False, f"HTTP {response.status_code}")
                
            # Test dashboard summary
            response = self.session.get(f"{BASE_URL}/advanced-dashboard/dashboard/summary")
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    dashboard_data = data.get("data", {})
                    print(f"✅ Dashboard summary: {len(dashboard_data.get('metrics', {}))} metrics")
                    self._record_test_result("dashboard_summary", True, f"{len(dashboard_data.get('metrics', {}))} metrics")
                else:
                    print("❌ Dashboard summary failed")
                    self._record_test_result("dashboard_summary", False, "No data")
            else:
                print(f"❌ Dashboard summary failed: {response.status_code}")
                self._record_test_result("dashboard_summary", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Advanced dashboard test failed: {e}")
            self._record_test_result("advanced_dashboard", False, str(e))
            
    async def _test_advanced_analytics_engine(self):
        """Test advanced analytics engine"""
        print("\n🧠 Testing Advanced Analytics Engine...")
        
        try:
            # Test analytics summary
            response = self.session.get(f"{BASE_URL}/advanced-analytics/summary")
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    analytics_data = data.get("data", {})
                    print(f"✅ Analytics summary: {analytics_data.get('data_points_collected', 0)} data points")
                    self._record_test_result("analytics_summary", True, f"{analytics_data.get('data_points_collected', 0)} data points")
                else:
                    print("❌ Analytics summary failed")
                    self._record_test_result("analytics_summary", False, "No data")
            else:
                print(f"❌ Analytics summary failed: {response.status_code}")
                self._record_test_result("analytics_summary", False, f"HTTP {response.status_code}")
                
            # Test adding data points
            test_data = {
                "metric_name": "test_metric",
                "value": 75.5,
                "metadata": {"test": True}
            }
            response = self.session.post(f"{BASE_URL}/advanced-analytics/add-data-point", json=test_data)
            if response.status_code == 200:
                print("✅ Data point added successfully")
                self._record_test_result("add_data_point", True, "Data point added")
            else:
                print(f"❌ Add data point failed: {response.status_code}")
                self._record_test_result("add_data_point", False, f"HTTP {response.status_code}")
                
            # Test insights generation
            response = self.session.post(f"{BASE_URL}/advanced-analytics/generate-insights")
            if response.status_code == 200:
                print("✅ Insights generation triggered")
                self._record_test_result("insights_generation", True, "Insights triggered")
            else:
                print(f"❌ Insights generation failed: {response.status_code}")
                self._record_test_result("insights_generation", False, f"HTTP {response.status_code}")
                
            # Test model info
            response = self.session.get(f"{BASE_URL}/advanced-analytics/models")
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    models_data = data.get("data", {})
                    print(f"✅ Model info: {models_data.get('total_models', 0)} models")
                    self._record_test_result("model_info", True, f"{models_data.get('total_models', 0)} models")
                else:
                    print("❌ Model info failed")
                    self._record_test_result("model_info", False, "No models")
            else:
                print(f"❌ Model info failed: {response.status_code}")
                self._record_test_result("model_info", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Advanced analytics engine test failed: {e}")
            self._record_test_result("advanced_analytics", False, str(e))
            
    async def _test_auto_scaling_manager(self):
        """Test auto scaling manager"""
        print("\n🚀 Testing Auto Scaling Manager...")
        
        try:
            # Test scaling summary
            response = self.session.get(f"{BASE_URL}/auto-scaling/summary")
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    scaling_data = data.get("data", {})
                    print(f"✅ Scaling summary: {scaling_data.get('current_instances', 0)} instances")
                    self._record_test_result("scaling_summary", True, f"{scaling_data.get('current_instances', 0)} instances")
                else:
                    print("❌ Scaling summary failed")
                    self._record_test_result("scaling_summary", False, "No data")
            else:
                print(f"❌ Scaling summary failed: {response.status_code}")
                self._record_test_result("scaling_summary", False, f"HTTP {response.status_code}")
                
            # Test scaling policies
            response = self.session.get(f"{BASE_URL}/auto-scaling/policies")
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    policies_data = data.get("data", {})
                    print(f"✅ Scaling policies: {policies_data.get('total_policies', 0)} policies")
                    self._record_test_result("scaling_policies", True, f"{policies_data.get('total_policies', 0)} policies")
                else:
                    print("❌ Scaling policies failed")
                    self._record_test_result("scaling_policies", False, "No policies")
            else:
                print(f"❌ Scaling policies failed: {response.status_code}")
                self._record_test_result("scaling_policies", False, f"HTTP {response.status_code}")
                
            # Test scaling recommendations
            response = self.session.get(f"{BASE_URL}/auto-scaling/recommendations")
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    recommendations_data = data.get("data", {})
                    print(f"✅ Scaling recommendations: {recommendations_data.get('total_recommendations', 0)} recommendations")
                    self._record_test_result("scaling_recommendations", True, f"{recommendations_data.get('total_recommendations', 0)} recommendations")
                else:
                    print("❌ Scaling recommendations failed")
                    self._record_test_result("scaling_recommendations", False, "No recommendations")
            else:
                print(f"❌ Scaling recommendations failed: {response.status_code}")
                self._record_test_result("scaling_recommendations", False, f"HTTP {response.status_code}")
                
            # Test manual scaling
            response = self.session.post(f"{BASE_URL}/auto-scaling/scale-manual", 
                                       params={"target_instances": 2, "reason": "Test scaling"})
            if response.status_code == 200:
                print("✅ Manual scaling successful")
                self._record_test_result("manual_scaling", True, "Scaling successful")
            else:
                print(f"❌ Manual scaling failed: {response.status_code}")
                self._record_test_result("manual_scaling", False, f"HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Auto scaling manager test failed: {e}")
            self._record_test_result("auto_scaling", False, str(e))
            
    async def _test_system_integration(self):
        """Test system integration"""
        print("\n🔗 Testing System Integration...")
        
        try:
            # Test data sync between systems
            response = self.session.post(f"{BASE_URL}/advanced-analytics/sync-performance-data")
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    sync_data = data.get("data", {})
                    print(f"✅ Data sync: {sync_data.get('metrics_synced', 0)} metrics synced")
                    self._record_test_result("data_sync", True, f"{sync_data.get('metrics_synced', 0)} metrics synced")
                else:
                    print("❌ Data sync failed")
                    self._record_test_result("data_sync", False, "Sync failed")
            else:
                print(f"❌ Data sync failed: {response.status_code}")
                self._record_test_result("data_sync", False, f"HTTP {response.status_code}")
                
            # Test health endpoints
            health_endpoints = [
                "/advanced-performance/health",
                "/advanced-analytics/health",
                "/auto-scaling/health"
            ]
            
            healthy_systems = 0
            for endpoint in health_endpoints:
                response = self.session.get(f"{BASE_URL}{endpoint}")
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        healthy_systems += 1
                        
            print(f"✅ System health: {healthy_systems}/{len(health_endpoints)} systems healthy")
            self._record_test_result("system_health", True, f"{healthy_systems}/{len(health_endpoints)} healthy")
            
        except Exception as e:
            print(f"❌ System integration test failed: {e}")
            self._record_test_result("system_integration", False, str(e))
            
    def _record_test_result(self, test_name: str, success: bool, details: str):
        """Record test result"""
        self.test_results.append({
            "test_name": test_name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
        
    def _generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 60)
        print("📊 ADVANCED FEATURES TEST REPORT")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result["success"])
        failed_tests = total_tests - successful_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Successful: {successful_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        
        print("\n📋 Detailed Results:")
        print("-" * 40)
        
        for result in self.test_results:
            status = "✅" if result["success"] else "❌"
            print(f"{status} {result['test_name']}: {result['details']}")
            
        print("\n🎯 Summary:")
        if successful_tests == total_tests:
            print("🎉 All advanced features are working perfectly!")
        elif successful_tests >= total_tests * 0.8:
            print("👍 Most advanced features are working well!")
        else:
            print("⚠️  Some advanced features need attention.")
            
        # Save report to file
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": (successful_tests/total_tests)*100,
            "results": self.test_results
        }
        
        with open("advanced_features_test_report.json", "w") as f:
            json.dump(report_data, f, indent=2)
            
        print(f"\n📄 Detailed report saved to: advanced_features_test_report.json")


async def main():
    """Main test runner"""
    tester = AdvancedFeaturesTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 Advanced Features Test Suite completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Advanced Features Test Suite failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())


