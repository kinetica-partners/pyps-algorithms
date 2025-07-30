#!/usr/bin/env python3
"""
Comprehensive analysis script to identify redundant code and prepare cleanup plan.
This script will:
1. Scan for redundant code patterns
2. Check test coverage
3. Create a cleanup plan with tests for each change
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set

def analyze_redundant_code():
    """Analyze the forecast_ets.py file for redundant code patterns."""
    print("🔍 COMPREHENSIVE CODE ANALYSIS")
    print("=" * 80)
    
    with open('src/forecast_ets.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    issues = []
    
    print("📋 SCANNING FOR REDUNDANT CODE PATTERNS...")
    print("-" * 50)
    
    # 1. Non-pythonic sys.path.append patterns
    sys_path_issues = []
    for i, line in enumerate(lines):
        if 'sys.path.append(str(Path(__file__).parent' in line:
            sys_path_issues.append({
                'line': i + 1,
                'content': line.strip(),
                'issue': 'Non-pythonic sys.path.append with Path conversion'
            })
    
    if sys_path_issues:
        print("❌ ISSUE 1: Non-pythonic sys.path.append patterns")
        for issue in sys_path_issues:
            print(f"   Line {issue['line']}: {issue['content']}")
        print("   💡 Should use relative imports or proper package structure")
    
    # 2. Error suppression that doesn't work
    error_suppress_issues = []
    in_try_block = False
    try_line = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('try:'):
            in_try_block = True
            try_line = i + 1
        elif stripped.startswith('except:') and in_try_block:
            # Look for bare except or ineffective suppression
            if stripped == 'except:' or 'pass' in lines[i+1] if i+1 < len(lines) else False:
                error_suppress_issues.append({
                    'try_line': try_line,
                    'except_line': i + 1,
                    'content': stripped,
                    'issue': 'Bare except or ineffective error suppression'
                })
            in_try_block = False
    
    if error_suppress_issues:
        print("\n❌ ISSUE 2: Ineffective error suppression")
        for issue in error_suppress_issues:
            print(f"   Lines {issue['try_line']}-{issue['except_line']}: {issue['content']}")
        print("   💡 Should use specific exception types and proper logging")
    
    # 3. Duplicate imports
    import_lines = []
    for i, line in enumerate(lines):
        if line.strip().startswith(('import ', 'from ')) and not line.strip().startswith('#'):
            import_lines.append((i + 1, line.strip()))
    
    seen_imports = set()
    duplicate_imports = []
    for line_num, import_line in import_lines:
        if import_line in seen_imports:
            duplicate_imports.append((line_num, import_line))
        seen_imports.add(import_line)
    
    if duplicate_imports:
        print("\n❌ ISSUE 3: Duplicate imports")
        for line_num, import_line in duplicate_imports:
            print(f"   Line {line_num}: {import_line}")
    
    # 4. Unused imports and functions
    print("\n🔍 SCANNING FOR UNUSED CODE...")
    
    # Extract all function definitions
    function_defs = re.findall(r'def (\w+)\(', content)
    function_calls = set()
    
    # Find all function calls in the code
    for match in re.finditer(r'(\w+)\s*\(', content):
        func_name = match.group(1)
        if func_name not in ['if', 'for', 'while', 'try', 'except', 'with', 'print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple']:
            function_calls.add(func_name)
    
    # Find potentially unused functions (defined but not called)
    unused_functions = []
    for func in function_defs:
        if func not in function_calls and func != 'main':  # main is special
            unused_functions.append(func)
    
    if unused_functions:
        print("⚠️  POTENTIALLY UNUSED FUNCTIONS:")
        for func in unused_functions:
            print(f"   - {func}")
        print("   💡 Verify these are actually unused before removing")
    
    # 5. Redundant string formatting
    redundant_formatting = []
    for i, line in enumerate(lines):
        # Look for f-strings that could be simplified
        if 'f"' in line and '{' in line and '}' in line:
            # Check for f"{variable}" patterns that could be just str(variable)
            simple_fstring = re.search(r'f"{\s*(\w+)\s*}"', line)
            if simple_fstring:
                redundant_formatting.append({
                    'line': i + 1,
                    'content': line.strip(),
                    'suggestion': f"Use str({simple_fstring.group(1)}) instead"
                })
    
    if redundant_formatting:
        print("\n⚠️  REDUNDANT STRING FORMATTING:")
        for issue in redundant_formatting[:5]:  # Show first 5
            print(f"   Line {issue['line']}: {issue['suggestion']}")
    
    # 6. Dead code patterns
    dead_code_patterns = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Look for commented-out code that looks like old implementations
        if stripped.startswith('#') and any(pattern in stripped for pattern in ['def ', 'import ', 'return ', 'if __name__']):
            dead_code_patterns.append(i + 1)
    
    if dead_code_patterns:
        print(f"\n⚠️  POTENTIAL DEAD CODE: {len(dead_code_patterns)} lines of commented code")
        print("   💡 Review and remove obsolete commented code")
    
    # 7. Magic numbers and hardcoded values
    magic_numbers = []
    for i, line in enumerate(lines):
        # Look for hardcoded numbers that should be constants
        numbers = re.findall(r'\b(\d{2,})\b', line)
        for num in numbers:
            if int(num) > 10 and int(num) not in [52, 104, 26, 12, 24]:  # Common valid values
                if not any(pattern in line.lower() for pattern in ['week', 'month', 'year', 'period', 'season']):
                    magic_numbers.append({
                        'line': i + 1,
                        'number': num,
                        'content': line.strip()[:60] + '...' if len(line.strip()) > 60 else line.strip()
                    })
    
    if magic_numbers:
        print(f"\n⚠️  MAGIC NUMBERS: Found {len(magic_numbers)} potential magic numbers")
        for issue in magic_numbers[:3]:  # Show first 3
            print(f"   Line {issue['line']}: {issue['number']} in '{issue['content']}'")
    
    return {
        'sys_path_issues': sys_path_issues,
        'error_suppress_issues': error_suppress_issues,
        'duplicate_imports': duplicate_imports,
        'unused_functions': unused_functions,
        'redundant_formatting': redundant_formatting,
        'dead_code_patterns': dead_code_patterns,
        'magic_numbers': magic_numbers
    }

def check_test_coverage():
    """Check current test coverage."""
    print("\n🧪 CHECKING TEST COVERAGE")
    print("=" * 50)
    
    try:
        # Run pytest with coverage
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/test_forecast_ets.py', 
            '--cov=src.forecast_ets',
            '--cov-report=term-missing',
            '--tb=short'
        ], capture_output=True, text=True, timeout=120)
        
        print("📊 COVERAGE REPORT:")
        print(result.stdout)
        
        if result.stderr:
            print("\n⚠️  STDERR:")
            print(result.stderr)
        
        # Extract coverage percentage
        coverage_match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', result.stdout)
        if coverage_match:
            coverage = int(coverage_match.group(1))
            print(f"\n📈 Current Coverage: {coverage}%")
            return coverage
        else:
            print("\n❓ Could not determine coverage percentage")
            return None
            
    except subprocess.TimeoutExpired:
        print("❌ Test coverage check timed out")
        return None
    except Exception as e:
        print(f"❌ Error checking test coverage: {e}")
        return None

def analyze_test_structure():
    """Analyze test structure and identify gaps."""
    print("\n🔍 ANALYZING TEST STRUCTURE")
    print("=" * 50)
    
    with open('tests/test_forecast_ets.py', 'r', encoding='utf-8') as f:
        test_content = f.read()
    
    # Find all test functions
    test_functions = re.findall(r'def (test_\w+)', test_content)
    print(f"📋 Found {len(test_functions)} test functions")
    
    # Categorize tests
    categories = {
        'basic_functionality': [],
        'accuracy_validation': [],
        'input_validation': [],
        'edge_cases': [],
        'rolling_forecast': [],
        'multiseries': [],
        'other': []
    }
    
    for test_func in test_functions:
        if 'basic' in test_func or 'functionality' in test_func:
            categories['basic_functionality'].append(test_func)
        elif 'accuracy' in test_func or 'validation' in test_func:
            categories['accuracy_validation'].append(test_func)
        elif 'input' in test_func or 'validation' in test_func:
            categories['input_validation'].append(test_func)
        elif 'edge' in test_func or 'missing' in test_func:
            categories['edge_cases'].append(test_func)
        elif 'rolling' in test_func:
            categories['rolling_forecast'].append(test_func)
        elif 'multiseries' in test_func or 'multi' in test_func:
            categories['multiseries'].append(test_func)
        else:
            categories['other'].append(test_func)
    
    print("\n📊 Test Categories:")
    for category, tests in categories.items():
        print(f"   {category}: {len(tests)} tests")
    
    # Check for sys.path.append issues in tests
    test_sys_path_issues = []
    test_lines = test_content.split('\n')
    for i, line in enumerate(test_lines):
        if 'sys.path.append(str(Path(__file__).parent' in line:
            test_sys_path_issues.append(i + 1)
    
    if test_sys_path_issues:
        print(f"\n❌ Test file has {len(test_sys_path_issues)} non-pythonic sys.path.append issues")
    
    return {
        'test_functions': test_functions,
        'categories': categories,
        'test_sys_path_issues': test_sys_path_issues
    }

def create_cleanup_plan(analysis_results, test_analysis):
    """Create a comprehensive cleanup plan."""
    print("\n📋 CREATING CLEANUP PLAN")
    print("=" * 80)
    
    cleanup_tasks = []
    
    # Task 1: Fix sys.path.append issues
    if analysis_results['sys_path_issues'] or test_analysis['test_sys_path_issues']:
        cleanup_tasks.append({
            'priority': 'HIGH',
            'task': 'Fix non-pythonic sys.path.append patterns',
            'description': 'Replace str(Path(__file__).parent / "data_generators") with proper relative imports',
            'files': ['src/forecast_ets.py', 'tests/test_forecast_ets.py'],
            'test_required': True,
            'risk': 'MEDIUM'
        })
    
    # Task 2: Remove redundant statsforecast code
    cleanup_tasks.append({
        'priority': 'HIGH',
        'task': 'Remove unused generate_forecast_ets_statsforecast function',
        'description': 'Function is no longer used after refactoring, but should verify all imports',
        'files': ['src/forecast_ets.py'],
        'test_required': True,
        'risk': 'MEDIUM'
    })
    
    # Task 3: Fix error suppression
    if analysis_results['error_suppress_issues']:
        cleanup_tasks.append({
            'priority': 'MEDIUM',
            'task': 'Improve error handling',
            'description': 'Replace bare except clauses with specific exception types',
            'files': ['src/forecast_ets.py'],
            'test_required': True,
            'risk': 'HIGH'
        })
    
    # Task 4: Remove duplicate imports
    if analysis_results['duplicate_imports']:
        cleanup_tasks.append({
            'priority': 'LOW',
            'task': 'Remove duplicate imports',
            'description': 'Clean up duplicate import statements',
            'files': ['src/forecast_ets.py'],
            'test_required': False,
            'risk': 'LOW'
        })
    
    # Task 5: Review unused functions
    if analysis_results['unused_functions']:
        cleanup_tasks.append({
            'priority': 'MEDIUM',
            'task': 'Review and remove unused functions',
            'description': f"Review functions: {', '.join(analysis_results['unused_functions'])}",
            'files': ['src/forecast_ets.py'],
            'test_required': True,
            'risk': 'MEDIUM'
        })
    
    # Task 6: Clean up dead code
    if analysis_results['dead_code_patterns']:
        cleanup_tasks.append({
            'priority': 'LOW',
            'task': 'Remove commented dead code',
            'description': f"Remove {len(analysis_results['dead_code_patterns'])} lines of commented code",
            'files': ['src/forecast_ets.py'],
            'test_required': False,
            'risk': 'LOW'
        })
    
    # Task 7: Extract magic numbers to constants
    if analysis_results['magic_numbers']:
        cleanup_tasks.append({
            'priority': 'MEDIUM',
            'task': 'Extract magic numbers to constants',
            'description': f"Extract {len(analysis_results['magic_numbers'])} magic numbers to named constants",
            'files': ['src/forecast_ets.py'],
            'test_required': True,
            'risk': 'LOW'
        })
    
    print(f"📊 Generated {len(cleanup_tasks)} cleanup tasks")
    print()
    
    for i, task in enumerate(cleanup_tasks, 1):
        print(f"{i}. [{task['priority']}] {task['task']}")
        print(f"   📝 {task['description']}")
        print(f"   📁 Files: {', '.join(task['files'])}")
        print(f"   🧪 Test required: {'Yes' if task['test_required'] else 'No'}")
        print(f"   ⚠️  Risk: {task['risk']}")
        print()
    
    return cleanup_tasks

def run_baseline_tests():
    """Run baseline tests before cleanup."""
    print("🧪 RUNNING BASELINE TESTS")
    print("=" * 50)
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            'tests/test_forecast_ets.py', 
            '-v',
            '--tb=short'
        ], capture_output=True, text=True, timeout=180)
        
        print("📊 BASELINE TEST RESULTS:")
        # Extract summary
        lines = result.stdout.split('\n')
        for line in lines:
            if any(keyword in line for keyword in ['passed', 'failed', 'error', 'FAILED', 'PASSED', '======']):
                print(line)
        
        if result.stderr:
            print("\n⚠️  STDERR:")
            print(result.stderr[-500:])  # Last 500 chars
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Baseline tests timed out")
        return False
    except Exception as e:
        print(f"❌ Error running baseline tests: {e}")
        return False

def main():
    """Main analysis function."""
    print("🔍 COMPREHENSIVE FORECAST_ETS.PY CLEANUP ANALYSIS")
    print("=" * 80)
    
    # Step 1: Analyze redundant code
    analysis_results = analyze_redundant_code()
    
    # Step 2: Analyze test structure
    test_analysis = analyze_test_structure()
    
    # Step 3: Run baseline tests
    baseline_passed = run_baseline_tests()
    
    # Step 4: Check test coverage
    coverage = check_test_coverage()
    
    # Step 5: Create cleanup plan
    cleanup_tasks = create_cleanup_plan(analysis_results, test_analysis)
    
    # Step 6: Summary and recommendations
    print("📋 ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"📊 Code Issues Found:")
    print(f"   • sys.path.append issues: {len(analysis_results['sys_path_issues'])}")
    print(f"   • Error suppression issues: {len(analysis_results['error_suppress_issues'])}")
    print(f"   • Duplicate imports: {len(analysis_results['duplicate_imports'])}")
    print(f"   • Unused functions: {len(analysis_results['unused_functions'])}")
    print(f"   • Magic numbers: {len(analysis_results['magic_numbers'])}")
    print(f"   • Dead code lines: {len(analysis_results['dead_code_patterns'])}")
    
    print(f"\n🧪 Test Analysis:")
    print(f"   • Total test functions: {len(test_analysis['test_functions'])}")
    print(f"   • Test coverage: {coverage}%" if coverage else "   • Test coverage: Unknown")
    print(f"   • Baseline tests: {'✅ PASSED' if baseline_passed else '❌ FAILED'}")
    
    print(f"\n📋 Cleanup Tasks: {len(cleanup_tasks)} tasks identified")
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Review cleanup plan above")
    print("2. Start with HIGH priority tasks")
    print("3. Run tests after each change")
    print("4. Ensure test coverage remains >90%")
    
    if not baseline_passed:
        print("\n⚠️  WARNING: Baseline tests are failing! Fix tests first before cleanup.")
    
    return {
        'analysis_results': analysis_results,
        'test_analysis': test_analysis,
        'cleanup_tasks': cleanup_tasks,
        'baseline_passed': baseline_passed,
        'coverage': coverage
    }

if __name__ == "__main__":
    results = main()