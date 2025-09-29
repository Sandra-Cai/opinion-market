#!/usr/bin/env python3
"""
Advanced Test Runner
Comprehensive test execution for advanced features
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class AdvancedTestRunner:
    """Advanced test runner with comprehensive reporting"""
    
    def __init__(self):
        self.project_root = project_root
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all advanced feature tests"""
        print("🚀 Starting Advanced Feature Test Suite")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Define test categories
        test_categories = {
            "advanced_analytics": {
                "file": "tests/test_advanced_analytics.py",
                "description": "Advanced Analytics Engine Tests",
                "tags": ["analytics", "ml", "predictions"]
            },
            "auto_scaling": {
                "file": "tests/test_auto_scaling.py", 
                "description": "Auto Scaling Manager Tests",
                "tags": ["scaling", "performance", "automation"]
            },
            "advanced_dashboard": {
                "file": "tests/test_advanced_dashboard.py",
                "description": "Advanced Dashboard Tests", 
                "tags": ["dashboard", "websocket", "realtime"]
            },
            "performance_tests": {
                "file": "tests/performance/test_performance.py",
                "description": "Performance Tests",
                "tags": ["performance", "load", "benchmark"]
            }
        }
        
        # Run each test category
        for category, config in test_categories.items():
            print(f"\n📊 Running {config['description']}...")
            result = self.run_test_category(category, config)
            self.test_results[category] = result
            
        self.end_time = time.time()
        
        # Generate comprehensive report
        report = self.generate_test_report()
        
        # Save report
        self.save_test_report(report)
        
        return report
    
    def run_test_category(self, category: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific test category"""
        test_file = self.project_root / config["file"]
        
        if not test_file.exists():
            return {
                "status": "skipped",
                "reason": f"Test file not found: {test_file}",
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
                "duration": 0
            }
        
        try:
            # Run pytest with detailed output
            cmd = [
                sys.executable, "-m", "pytest",
                str(test_file),
                "-v",
                "--tb=short",
                "--durations=10",
                "--json-report",
                "--json-report-file=/tmp/pytest_report.json"
            ]
            
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
            "categories": self.test_results,
            "recommendations": self.generate_recommendations()
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
        
        # Check for specific categories with issues
        for category, result in self.test_results.items():
            if result.get("status") == "error":
                recommendations.append(f"🚨 Fix {category} test execution errors")
            elif result.get("tests_failed", 0) > 0:
                recommendations.append(f"🔍 Review {category} test failures")
        
        if not recommendations:
            recommendations.append("✅ All tests are passing - great job!")
            recommendations.append("🚀 Consider adding more edge case tests")
            recommendations.append("📊 Add performance benchmarks for critical paths")
        
        return recommendations
    
    def save_test_report(self, report: Dict[str, Any]):
        """Save test report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.project_root / f"test_report_advanced_{timestamp}.json"
        
        try:
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Test report saved to: {report_file}")
        except Exception as e:
            print(f"❌ Failed to save test report: {e}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print test summary to console"""
        print("\n" + "=" * 60)
        print("📊 ADVANCED FEATURE TEST SUMMARY")
        print("=" * 60)
        
        summary = report["summary"]
        print(f"Overall Status: {report['overall_status']}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['tests_passed']} ✅")
        print(f"Failed: {summary['tests_failed']} ❌")
        print(f"Skipped: {summary['tests_skipped']} ⏭️")
        print(f"Success Rate: {summary['success_rate']}%")
        print(f"Total Duration: {summary['total_duration']}s")
        
        print("\n📋 Category Breakdown:")
        for category, result in report["categories"].items():
            status_icon = "✅" if result.get("tests_failed", 0) == 0 else "❌"
            print(f"  {status_icon} {category}: {result.get('tests_passed', 0)}/{result.get('tests_run', 0)} passed")
        
        print("\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"  {rec}")
        
        print("=" * 60)


def main():
    """Main entry point"""
    runner = AdvancedTestRunner()
    
    try:
        # Run all tests
        report = runner.run_all_tests()
        
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
