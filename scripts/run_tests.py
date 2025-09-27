#!/usr/bin/env python3
"""
Test Automation Script
Comprehensive test runner for Opinion Market platform
"""

import os
import sys
import asyncio
import argparse
import subprocess
import time
from typing import List, Dict, Any
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestRunner:
    """Comprehensive test runner for the Opinion Market platform"""
    
    def __init__(self):
        self.project_root = project_root
        self.test_results = {}
        self.start_time = time.time()
    
    def run_unit_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run unit tests"""
        print("🧪 Running Unit Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            "tests/unit/",
            "-v" if verbose else "-q",
            "--tb=short",
            "--maxfail=5",
            "--durations=10"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return {
            "type": "unit",
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    
    def run_integration_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run integration tests"""
        print("🔗 Running Integration Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            "tests/integration/",
            "-v" if verbose else "-q",
            "--tb=short",
            "--maxfail=3",
            "--durations=10"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return {
            "type": "integration",
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    
    def run_performance_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run performance tests"""
        print("⚡ Running Performance Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            "tests/performance/",
            "-v" if verbose else "-q",
            "--tb=short",
            "--maxfail=2",
            "--durations=10",
            "-m", "performance"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return {
            "type": "performance",
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    
    def run_api_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run API tests"""
        print("🌐 Running API Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            "tests/api/",
            "-v" if verbose else "-q",
            "--tb=short",
            "--maxfail=3",
            "--durations=10"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return {
            "type": "api",
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    
    def run_security_tests(self, verbose: bool = False) -> Dict[str, Any]:
        """Run security tests"""
        print("🔒 Running Security Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            "tests/security/",
            "-v" if verbose else "-q",
            "--tb=short",
            "--maxfail=2",
            "--durations=10"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return {
            "type": "security",
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    
    def run_coverage_analysis(self) -> Dict[str, Any]:
        """Run test coverage analysis"""
        print("📊 Running Coverage Analysis...")
        
        cmd = [
            "python", "-m", "pytest",
            "tests/",
            "--cov=app",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--cov-fail-under=80"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return {
            "type": "coverage",
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    
    def run_linting(self) -> Dict[str, Any]:
        """Run code linting"""
        print("🔍 Running Code Linting...")
        
        # Run flake8
        flake8_cmd = ["python", "-m", "flake8", "app/", "tests/", "--max-line-length=100"]
        flake8_result = subprocess.run(flake8_cmd, capture_output=True, text=True)
        
        # Run black check
        black_cmd = ["python", "-m", "black", "--check", "app/", "tests/"]
        black_result = subprocess.run(black_cmd, capture_output=True, text=True)
        
        # Run isort check
        isort_cmd = ["python", "-m", "isort", "--check-only", "app/", "tests/"]
        isort_result = subprocess.run(isort_cmd, capture_output=True, text=True)
        
        success = all([
            flake8_result.returncode == 0,
            black_result.returncode == 0,
            isort_result.returncode == 0
        ])
        
        return {
            "type": "linting",
            "success": success,
            "flake8": {
                "success": flake8_result.returncode == 0,
                "output": flake8_result.stdout + flake8_result.stderr
            },
            "black": {
                "success": black_result.returncode == 0,
                "output": black_result.stdout + black_result.stderr
            },
            "isort": {
                "success": isort_result.returncode == 0,
                "output": isort_result.stdout + isort_result.stderr
            }
        }
    
    def run_type_checking(self) -> Dict[str, Any]:
        """Run type checking with mypy"""
        print("🔬 Running Type Checking...")
        
        cmd = [
            "python", "-m", "mypy",
            "app/",
            "--ignore-missing-imports",
            "--no-strict-optional",
            "--warn-unused-ignores"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        return {
            "type": "type_checking",
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    
    def run_security_scanning(self) -> Dict[str, Any]:
        """Run security scanning"""
        print("🛡️ Running Security Scanning...")
        
        # Run bandit
        bandit_cmd = ["python", "-m", "bandit", "-r", "app/", "-f", "json"]
        bandit_result = subprocess.run(bandit_cmd, capture_output=True, text=True)
        
        # Run safety
        safety_cmd = ["python", "-m", "safety", "check", "--json"]
        safety_result = subprocess.run(safety_cmd, capture_output=True, text=True)
        
        success = bandit_result.returncode == 0 and safety_result.returncode == 0
        
        return {
            "type": "security_scanning",
            "success": success,
            "bandit": {
                "success": bandit_result.returncode == 0,
                "output": bandit_result.stdout + bandit_result.stderr
            },
            "safety": {
                "success": safety_result.returncode == 0,
                "output": safety_result.stdout + safety_result.stderr
            }
        }
    
    def run_all_tests(self, test_types: List[str], verbose: bool = False) -> Dict[str, Any]:
        """Run all specified test types"""
        results = {}
        
        for test_type in test_types:
            if test_type == "unit":
                results["unit"] = self.run_unit_tests(verbose)
            elif test_type == "integration":
                results["integration"] = self.run_integration_tests(verbose)
            elif test_type == "performance":
                results["performance"] = self.run_performance_tests(verbose)
            elif test_type == "api":
                results["api"] = self.run_api_tests(verbose)
            elif test_type == "security":
                results["security"] = self.run_security_tests(verbose)
            elif test_type == "coverage":
                results["coverage"] = self.run_coverage_analysis()
            elif test_type == "linting":
                results["linting"] = self.run_linting()
            elif test_type == "type_checking":
                results["type_checking"] = self.run_type_checking()
            elif test_type == "security_scanning":
                results["security_scanning"] = self.run_security_scanning()
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate test report"""
        total_time = time.time() - self.start_time
        
        report = []
        report.append("=" * 80)
        report.append("🧪 OPINION MARKET TEST REPORT")
        report.append("=" * 80)
        report.append(f"⏱️  Total Execution Time: {total_time:.2f} seconds")
        report.append("")
        
        # Summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.get("success", False))
        failed_tests = total_tests - passed_tests
        
        report.append("📊 SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Test Suites: {total_tests}")
        report.append(f"✅ Passed: {passed_tests}")
        report.append(f"❌ Failed: {failed_tests}")
        report.append(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "N/A")
        report.append("")
        
        # Detailed results
        report.append("📋 DETAILED RESULTS")
        report.append("-" * 40)
        
        for test_type, result in results.items():
            status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
            report.append(f"{test_type.upper()}: {status}")
            
            if not result.get("success", False):
                if "stdout" in result and result["stdout"]:
                    report.append(f"  Output: {result['stdout'][:200]}...")
                if "stderr" in result and result["stderr"]:
                    report.append(f"  Errors: {result['stderr'][:200]}...")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_report(self, report: str, filename: str = "test_report.txt"):
        """Save test report to file"""
        report_path = self.project_root / "reports" / filename
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, "w") as f:
            f.write(report)
        
        print(f"📄 Test report saved to: {report_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Opinion Market Test Runner")
    parser.add_argument(
        "--types",
        nargs="+",
        choices=[
            "unit", "integration", "performance", "api", "security",
            "coverage", "linting", "type_checking", "security_scanning"
        ],
        default=["unit", "integration", "coverage", "linting"],
        help="Test types to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate and save test report"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all test types"
    )
    
    args = parser.parse_args()
    
    # Determine test types
    if args.all:
        test_types = [
            "unit", "integration", "performance", "api", "security",
            "coverage", "linting", "type_checking", "security_scanning"
        ]
    else:
        test_types = args.types
    
    # Run tests
    runner = TestRunner()
    results = runner.run_all_tests(test_types, args.verbose)
    
    # Generate report
    report = runner.generate_report(results)
    print(report)
    
    # Save report if requested
    if args.report:
        runner.save_report(report)
    
    # Exit with appropriate code
    failed_tests = sum(1 for r in results.values() if not r.get("success", False))
    sys.exit(failed_tests)


if __name__ == "__main__":
    main()
