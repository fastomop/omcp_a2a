#!/usr/bin/env python3
"""
Test runner script for the A2A Medical Foundation Framework.

This script provides convenient ways to run different types of tests with appropriate configurations.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return the exit code."""
    print(f"🔄 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode == 0:
        print(f"✅ {description} - PASSED")
    else:
        print(f"❌ {description} - FAILED")
    
    print("-" * 60)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run tests for A2A Medical Foundation Framework")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "safety", "compliance", "fast"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--parallel", 
        action="store_true", 
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--failfast", 
        action="store_true", 
        help="Stop on first failure"
    )
    parser.add_argument(
        "--html-cov", 
        action="store_true", 
        help="Generate HTML coverage report"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test selection based on type
    if args.type == "unit":
        cmd.extend(["-m", "unit"])
    elif args.type == "integration":
        cmd.extend(["-m", "integration"])
    elif args.type == "safety":
        cmd.extend(["-m", "safety"])
    elif args.type == "compliance":
        cmd.extend(["-m", "compliance"])
    elif args.type == "fast":
        cmd.extend(["-m", "not slow"])
    # "all" runs everything (no marker filter)
    
    # Add coverage options
    if args.coverage or args.html_cov:
        cmd.extend([
            "--cov=src/a2a_medical",
            "--cov-report=term-missing"
        ])
        
        if args.html_cov:
            cmd.extend(["--cov-report=html"])
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    # Add verbose output
    if args.verbose:
        cmd.extend(["-v"])
    
    # Add fail fast
    if args.failfast:
        cmd.extend(["-x"])
    
    # Run the tests
    exit_code = run_command(cmd, f"Running {args.type} tests")
    
    if args.html_cov and exit_code == 0:
        print("\n📊 Coverage report generated at: htmlcov/index.html")
        print("To view: python -m http.server 8000 --directory htmlcov")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main()) 