#!/usr/bin/env python3
"""
Comprehensive Test Runner
Runs all test suites with detailed reporting and coverage analysis
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any


class TestRunner:
    """Comprehensive test runner with detailed reporting"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and generate comprehensive report"""
        print("🚀 Starting Comprehensive Test Suite")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Change to project directory
        os.chdir(self.project_root)
        
        # Run different test categories
        test_suites = [
            ("Unit Tests", self.run_unit_tests),
            ("Integration Tests", self.run_integration_tests),
            ("API Tests", self.run_api_tests),
            ("Performance Tests", self.run_performance_tests),
            ("Security Tests", self.run_security_tests),
            ("Coverage Analysis", self.run_coverage_analysis),
        ]
        
        for suite_name, test_func in test_suites:
            print(f"\n📋 Running {suite_name}...")
            print("-" * 40)
            
            try:
                result = test_func()
                self.test_results[suite_name] = result
                self.print_suite_result(suite_name, result)
            except Exception as e:
                error_result = {
                    "status": "error",
                    "error": str(e),
                    "duration": 0
                }
                self.test_results[suite_name] = error_result
                print(f"❌ {suite_name} failed: {e}")
        
        self.end_time = time.time()
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
        
        return self.test_results
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests"""
        start_time = time.time()
        
        cmd = [
            "python", "-m", "pytest", 
            "tests/unit/",
            "-v",
            "--tb=short",
            "--disable-warnings",
            "-m", "unit"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        duration = time.time() - start_time
        
        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": duration
        }
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests"""
        start_time = time.time()
        
        cmd = [
            "python", "-m", "pytest",
            "tests/integration/",
            "-v",
            "--tb=short",
            "--disable-warnings",
            "-m", "integration"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        duration = time.time() - start_time
        
        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": duration
        }
    
    def run_api_tests(self) -> Dict[str, Any]:
        """Run API tests"""
        start_time = time.time()
        
        cmd = [
            "python", "-m", "pytest",
            "tests/api/",
            "tests/test_api_integration.py",
            "tests/test_comprehensive_api.py",
            "-v",
            "--tb=short",
            "--disable-warnings",
            "-m", "api"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        duration = time.time() - start_time
        
        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": duration
        }
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests"""
        start_time = time.time()
        
        cmd = [
            "python", "-m", "pytest",
            "tests/test_performance_monitor.py",
            "tests/performance/",
            "-v",
            "--tb=short",
            "--disable-warnings",
            "-m", "not slow"  # Skip slow tests by default
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        duration = time.time() - start_time
        
        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": duration
        }
    
    def run_security_tests(self) -> Dict[str, Any]:
        """Run security tests"""
        start_time = time.time()
        
        cmd = [
            "python", "-m", "pytest",
            "tests/test_security_audit.py",
            "-v",
            "--tb=short",
            "--disable-warnings"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        duration = time.time() - start_time
        
        return {
            "status": "passed" if result.returncode == 0 else "failed",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": duration
        }
    
    def run_coverage_analysis(self) -> Dict[str, Any]:
        """Run coverage analysis"""
        start_time = time.time()
        
        cmd = [
            "python", "-m", "pytest",
            "tests/",
            "--cov=app",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-report=json:coverage.json",
            "--disable-warnings"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        duration = time.time() - start_time
        
        # Parse coverage data
        coverage_data = {}
        try:
            if os.path.exists("coverage.json"):
                with open("coverage.json", "r") as f:
                    coverage_data = json.load(f)
        except Exception as e:
            coverage_data = {"error": str(e)}
        
        return {
            "status": "completed",
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration": duration,
            "coverage_data": coverage_data
        }
    
    def print_suite_result(self, suite_name: str, result: Dict[str, Any]):
        """Print test suite result"""
        status_icon = "✅" if result["status"] in ["passed", "completed"] else "❌"
        duration = result.get("duration", 0)
        
        print(f"{status_icon} {suite_name}: {result['status']} ({duration:.2f}s)")
        
        if result["status"] == "failed" and result.get("stderr"):
            print(f"   Error: {result['stderr'][:200]}...")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive test report"""
        total_duration = self.end_time - self.start_time
        
        # Calculate summary statistics
        total_suites = len(self.test_results)
        passed_suites = sum(1 for result in self.test_results.values() 
                           if result["status"] in ["passed", "completed"])
        failed_suites = total_suites - passed_suites
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("=" * 60)
        print(f"Total Test Suites: {total_suites}")
        print(f"Passed: {passed_suites} ✅")
        print(f"Failed: {failed_suites} ❌")
        print(f"Total Duration: {total_duration:.2f} seconds")
        print(f"Success Rate: {(passed_suites/total_suites)*100:.1f}%")
        
        # Print detailed results
        print("\n📋 DETAILED RESULTS:")
        print("-" * 40)
        
        for suite_name, result in self.test_results.items():
            status_icon = "✅" if result["status"] in ["passed", "completed"] else "❌"
            duration = result.get("duration", 0)
            print(f"{status_icon} {suite_name:<20} {result['status']:<10} {duration:>6.2f}s")
        
        # Coverage summary
        if "Coverage Analysis" in self.test_results:
            coverage_result = self.test_results["Coverage Analysis"]
            if "coverage_data" in coverage_result and "totals" in coverage_result["coverage_data"]:
                totals = coverage_result["coverage_data"]["totals"]
                coverage_percent = totals.get("percent_covered", 0)
                print(f"\n📈 CODE COVERAGE: {coverage_percent:.1f}%")
        
        # Save report to file
        self.save_report_to_file()
        
        # Print recommendations
        self.print_recommendations()
    
    def save_report_to_file(self):
        """Save test report to file"""
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": self.end_time - self.start_time,
            "test_results": self.test_results,
            "summary": {
                "total_suites": len(self.test_results),
                "passed_suites": sum(1 for result in self.test_results.values() 
                                    if result["status"] in ["passed", "completed"]),
                "failed_suites": sum(1 for result in self.test_results.values() 
                                    if result["status"] == "failed")
            }
        }
        
        report_file = "test_report.json"
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
    
    def print_recommendations(self):
        """Print recommendations based on test results"""
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 40)
        
        failed_suites = [name for name, result in self.test_results.items() 
                        if result["status"] == "failed"]
        
        if failed_suites:
            print("🔧 Failed Test Suites:")
            for suite in failed_suites:
                print(f"   - {suite}: Review and fix failing tests")
        
        # Coverage recommendations
        if "Coverage Analysis" in self.test_results:
            coverage_result = self.test_results["Coverage Analysis"]
            if "coverage_data" in coverage_result and "totals" in coverage_result["coverage_data"]:
                totals = coverage_result["coverage_data"]["totals"]
                coverage_percent = totals.get("percent_covered", 0)
                
                if coverage_percent < 80:
                    print(f"📈 Low Coverage ({coverage_percent:.1f}%): Add more tests to improve coverage")
                elif coverage_percent < 90:
                    print(f"📈 Good Coverage ({coverage_percent:.1f}%): Consider adding tests for edge cases")
                else:
                    print(f"📈 Excellent Coverage ({coverage_percent:.1f}%)")
        
        # Performance recommendations
        if "Performance Tests" in self.test_results:
            perf_result = self.test_results["Performance Tests"]
            if perf_result["status"] == "failed":
                print("⚡ Performance Tests Failed: Review performance bottlenecks")
        
        # Security recommendations
        if "Security Tests" in self.test_results:
            security_result = self.test_results["Security Tests"]
            if security_result["status"] == "failed":
                print("🔒 Security Tests Failed: Review security vulnerabilities")
        
        print("\n🎯 Next Steps:")
        print("   1. Fix any failing tests")
        print("   2. Review code coverage and add missing tests")
        print("   3. Run security audit if needed")
        print("   4. Consider performance optimizations")


def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Comprehensive Test Runner")
        print("Usage: python run_comprehensive_tests.py [options]")
        print("\nOptions:")
        print("  --help     Show this help message")
        print("  --quick    Run only essential tests")
        print("  --full     Run all tests including slow ones")
        return
    
    runner = TestRunner()
    
    try:
        results = runner.run_all_tests()
        
        # Exit with appropriate code
        failed_suites = sum(1 for result in results.values() if result["status"] == "failed")
        sys.exit(failed_suites)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
