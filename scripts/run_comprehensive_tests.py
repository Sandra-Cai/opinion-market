#!/usr/bin/env python3
"""
Comprehensive Test Runner
Runs all tests including unit, integration, performance, and advanced feature tests
"""

import os
import sys
import subprocess
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class ComprehensiveTestRunner:
    """Comprehensive test runner with detailed reporting"""
    
    def __init__(self, verbose: bool = False, parallel: bool = False):
        self.project_root = project_root
        self.verbose = verbose
        self.parallel = parallel
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
    def run_all_tests(self, test_types: List[str] = None) -> Dict[str, Any]:
        """Run all tests or specified test types"""
        if test_types is None:
            test_types = ["unit", "integration", "performance", "advanced", "load"]
            
        print("🧪 Starting Comprehensive Test Suite")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Define test configurations
        test_configs = {
            "unit": {
                "pattern": "tests/test_*.py",
                "description": "Unit Tests",
                "timeout": 300
            },
            "integration": {
                "pattern": "tests/integration/test_*.py",
                "description": "Integration Tests",
                "timeout": 600
            },
            "performance": {
                "pattern": "tests/performance/test_*.py",
                "description": "Performance Tests",
                "timeout": 900
            },
            "advanced": {
                "pattern": "tests/test_advanced_*.py",
                "description": "Advanced Feature Tests",
                "timeout": 600
            },
            "load": {
                "pattern": "tests/load_test.py",
                "description": "Load Tests",
                "timeout": 1800
            }
        }
        
        # Run each test type
        for test_type in test_types:
            if test_type in test_configs:
                print(f"\n📊 Running {test_configs[test_type]['description']}...")
                result = self.run_test_type(test_type, test_configs[test_type])
                self.test_results[test_type] = result
            else:
                print(f"⚠️  Unknown test type: {test_type}")
                
        self.end_time = time.time()
        
        # Generate comprehensive report
        report = self.generate_test_report()
        
        # Save report
        self.save_test_report(report)
        
        return report
    
    def run_test_type(self, test_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific test type"""
        try:
            # Find test files
            test_files = list(self.project_root.glob(config["pattern"]))
            
            if not test_files:
                return {
                    "status": "skipped",
                    "reason": f"No test files found matching pattern: {config['pattern']}",
                    "tests_run": 0,
                    "tests_passed": 0,
                    "tests_failed": 0,
                    "duration": 0
                }
            
            # Run pytest with detailed output
            cmd = [
                sys.executable, "-m", "pytest",
                *[str(f) for f in test_files],
                "-v",
                "--tb=short",
                "--durations=10",
                "--json-report",
                "--json-report-file=/tmp/pytest_report.json",
                f"--timeout={config['timeout']}"
            ]
            
            if self.verbose:
                cmd.append("-s")
                
            if self.parallel and test_type != "load":
                cmd.extend(["-n", "auto"])
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            duration = time.time() - start_time
            
            # Parse pytest JSON report if available
            pytest_report = {}
            if os.path.exists("/tmp/pytest_report.json"):
                try:
                    with open("/tmp/pytest_report.json", "r") as f:
                        pytest_report = json.load(f)
                except:
                    pass
            
            return {
                "status": "completed",
                "tests_run": pytest_report.get("summary", {}).get("total", 0),
                "tests_passed": pytest_report.get("summary", {}).get("passed", 0),
                "tests_failed": pytest_report.get("summary", {}).get("failed", 0),
                "tests_skipped": pytest_report.get("summary", {}).get("skipped", 0),
                "duration": duration,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "pytest_report": pytest_report
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "duration": 0
            }
    
    def run_advanced_feature_tests(self) -> Dict[str, Any]:
        """Run advanced feature tests using the dedicated script"""
        try:
            cmd = [sys.executable, str(self.project_root / "scripts" / "run_advanced_tests.py")]
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            duration = time.time() - start_time
            
            return {
                "status": "completed",
                "duration": duration,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "duration": 0
            }
    
    def run_load_tests(self) -> Dict[str, Any]:
        """Run load tests using Locust"""
        try:
            # Check if Locust is available
            try:
                import locust
            except ImportError:
                return {
                    "status": "skipped",
                    "reason": "Locust not installed. Install with: pip install locust",
                    "duration": 0
                }
            
            # Start the application in background
            app_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", "app.main:app", 
                "--host", "0.0.0.0", "--port", "8000"
            ], cwd=self.project_root)
            
            # Wait for app to start
            time.sleep(30)
            
            # Run load tests
            cmd = [
                "locust",
                "-f", "tests/load_test.py",
                "--host=http://localhost:8000",
                "--users=50",
                "--spawn-rate=5",
                "--run-time=2m",
                "--headless",
                "--html=load_test_report.html"
            ]
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            duration = time.time() - start_time
            
            # Stop the application
            app_process.terminate()
            app_process.wait()
            
            return {
                "status": "completed",
                "duration": duration,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "duration": 0
            }
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = sum(r.get("tests_run", 0) for r in self.test_results.values())
        total_passed = sum(r.get("tests_passed", 0) for r in self.test_results.values())
        total_failed = sum(r.get("tests_failed", 0) for r in self.test_results.values())
        total_skipped = sum(r.get("tests_skipped", 0) for r in self.test_results.values())
        total_duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Calculate success rate
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        # Determine overall status
        if total_failed == 0:
            overall_status = "PASSED" if total_passed > 0 else "NO_TESTS"
        else:
            overall_status = "FAILED"
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "summary": {
                "total_tests": total_tests,
                "tests_passed": total_passed,
                "tests_failed": total_failed,
                "tests_skipped": total_skipped,
                "success_rate": round(success_rate, 2),
                "total_duration": round(total_duration, 2)
            },
            "test_types": self.test_results,
            "recommendations": self.generate_recommendations(),
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "project_root": str(self.project_root)
            }
        }
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        total_failed = sum(r.get("tests_failed", 0) for r in self.test_results.values())
        total_tests = sum(r.get("tests_run", 0) for r in self.test_results.values())
        
        if total_failed > 0:
            recommendations.append("🔧 Fix failing tests to improve code quality")
            recommendations.append("📝 Review test failures and update test cases as needed")
        
        if total_tests == 0:
            recommendations.append("⚠️ No tests were executed - check test file paths")
        
        # Check for specific test types with issues
        for test_type, result in self.test_results.items():
            if result.get("status") == "error":
                recommendations.append(f"🚨 Fix {test_type} test execution errors")
            elif result.get("tests_failed", 0) > 0:
                recommendations.append(f"🔍 Review {test_type} test failures")
        
        # Performance recommendations
        if "performance" in self.test_results:
            perf_result = self.test_results["performance"]
            if perf_result.get("tests_failed", 0) > 0:
                recommendations.append("⚡ Review performance test failures and optimize bottlenecks")
        
        # Advanced features recommendations
        if "advanced" in self.test_results:
            adv_result = self.test_results["advanced"]
            if adv_result.get("tests_failed", 0) > 0:
                recommendations.append("🚀 Review advanced feature test failures")
        
        if not recommendations:
            recommendations.append("✅ All tests are passing - great job!")
            recommendations.append("🚀 Consider adding more edge case tests")
            recommendations.append("📊 Add performance benchmarks for critical paths")
            recommendations.append("🔒 Add security tests for authentication and authorization")
        
        return recommendations
    
    def save_test_report(self, report: Dict[str, Any]):
        """Save test report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.project_root / f"test_report_comprehensive_{timestamp}.json"
        
        try:
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Test report saved to: {report_file}")
        except Exception as e:
            print(f"❌ Failed to save test report: {e}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print test summary to console"""
        print("\n" + "=" * 60)
        print("🧪 COMPREHENSIVE TEST SUMMARY")
        print("=" * 60)
        
        summary = report["summary"]
        print(f"Overall Status: {report['overall_status']}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['tests_passed']} ✅")
        print(f"Failed: {summary['tests_failed']} ❌")
        print(f"Skipped: {summary['tests_skipped']} ⏭️")
        print(f"Success Rate: {summary['success_rate']}%")
        print(f"Total Duration: {summary['total_duration']}s")
        
        print("\n📋 Test Type Breakdown:")
        for test_type, result in report["test_types"].items():
            status_icon = "✅" if result.get("tests_failed", 0) == 0 else "❌"
            tests_run = result.get("tests_run", 0)
            tests_passed = result.get("tests_passed", 0)
            duration = result.get("duration", 0)
            print(f"  {status_icon} {test_type}: {tests_passed}/{tests_run} passed ({duration:.1f}s)")
        
        print("\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"  {rec}")
        
        print("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Comprehensive Test Runner")
    parser.add_argument("--test-types", nargs="+", 
                       choices=["unit", "integration", "performance", "advanced", "load"],
                       help="Specific test types to run")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    parser.add_argument("--parallel", "-p", action="store_true", 
                       help="Run tests in parallel")
    parser.add_argument("--quick", action="store_true", 
                       help="Run only unit and integration tests")
    
    args = parser.parse_args()
    
    # Determine test types to run
    if args.quick:
        test_types = ["unit", "integration"]
    elif args.test_types:
        test_types = args.test_types
    else:
        test_types = None  # Run all tests
    
    runner = ComprehensiveTestRunner(verbose=args.verbose, parallel=args.parallel)
    
    try:
        # Run tests
        report = runner.run_all_tests(test_types)
        
        # Print summary
        runner.print_summary(report)
        
        # Exit with appropriate code
        if report["overall_status"] == "PASSED":
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()