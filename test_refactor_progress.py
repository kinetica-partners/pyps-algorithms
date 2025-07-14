import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import time
import ast
import inspect
from typing import Dict, List, Tuple, Any

# Add the src directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from working_calendar import (
    load_calendar_rules, load_calendar_exceptions, build_working_intervals, add_working_minutes
)
from test_working_calendar import WorkingCalendarTester

class SimplicityMetrics:
    """Calculate simplicity metrics for code analysis."""
    
    @staticmethod
    def count_lines_of_code(file_path: str) -> Dict[str, int]:
        """Count lines of code, excluding comments and blank lines."""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        blank_lines = sum(1 for line in lines if line.strip() == '')
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        code_lines = total_lines - blank_lines - comment_lines
        
        return {
            'total_lines': total_lines,
            'code_lines': code_lines,
            'blank_lines': blank_lines,
            'comment_lines': comment_lines
        }
    
    @staticmethod
    def count_functions_and_classes(file_path: str) -> Dict[str, int]:
        """Count number of functions and classes."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
            functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            return {'functions': functions, 'classes': classes}
        except:
            return {'functions': 0, 'classes': 0}
    
    @staticmethod
    def analyze_function_complexity(file_path: str) -> Dict[str, Any]:
        """Analyze function complexity metrics."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
            function_lengths = []
            function_params = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Count lines in function (approximate)
                    if (hasattr(node, 'end_lineno') and hasattr(node, 'lineno') and
                        node.end_lineno is not None and node.lineno is not None):
                        func_lines = node.end_lineno - node.lineno + 1
                    else:
                        func_lines = 10  # Default estimate
                    function_lengths.append(func_lines)
                    
                    # Count parameters
                    param_count = len(node.args.args)
                    function_params.append(param_count)
            
            return {
                'avg_function_length': sum(function_lengths) / len(function_lengths) if function_lengths else 0,
                'max_function_length': max(function_lengths) if function_lengths else 0,
                'avg_function_params': sum(function_params) / len(function_params) if function_params else 0,
                'max_function_params': max(function_params) if function_params else 0
            }
        except:
            return {
                'avg_function_length': 0,
                'max_function_length': 0,
                'avg_function_params': 0,
                'max_function_params': 0
            }

class PerformanceMetrics:
    """Measure performance metrics."""
    
    def __init__(self):
        self.results = {}
    
    def time_function(self, func_name: str, func, *args, **kwargs):
        """Time a function execution."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        self.results[func_name] = execution_time
        
        return result, execution_time
    
    def get_results(self) -> Dict[str, float]:
        """Get all timing results."""
        return self.results.copy()

class RefactorProgressTracker:
    """Track progress during refactoring."""
    
    def __init__(self, target_file: str):
        self.target_file = target_file
        self.simplicity = SimplicityMetrics()
        self.performance = PerformanceMetrics()
        
    def measure_current_state(self) -> Dict[str, Any]:
        """Measure current state of the code."""
        # Simplicity metrics
        loc_metrics = self.simplicity.count_lines_of_code(self.target_file)
        func_class_metrics = self.simplicity.count_functions_and_classes(self.target_file)
        complexity_metrics = self.simplicity.analyze_function_complexity(self.target_file)
        
        # Performance metrics
        perf_results = self._measure_performance()
        
        return {
            'simplicity': {
                **loc_metrics,
                **func_class_metrics,
                **complexity_metrics
            },
            'performance': perf_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def _measure_performance(self) -> Dict[str, float]:
        """Measure performance of current implementation."""
        # Load data
        rules_df = pd.read_csv('data/calendar_rules.csv')
        exceptions_df = pd.read_csv('data/calendar_exceptions.csv')
        
        # Time data loading
        start_time = time.perf_counter()
        rules = load_calendar_rules(rules_df)
        exceptions = load_calendar_exceptions(exceptions_df)
        data_loading_time = time.perf_counter() - start_time
        
        # Time interval building
        calendar_id = 'calC'
        calendar_start_date = datetime(2025, 1, 1).date()
        calendar_end_date = datetime(2025, 12, 31).date()
        
        start_time = time.perf_counter()
        working_intervals = build_working_intervals(
            rules, exceptions, calendar_id, calendar_start_date, calendar_end_date
        )
        interval_building_time = time.perf_counter() - start_time
        
        # Time multiple add_working_minutes operations
        test_cases = [
            (datetime(2025, 1, 6, 9, 0), 120),
            (datetime(2025, 1, 6, 11, 50), 60),
            (datetime(2025, 1, 1, 9, 0), 60),
            (datetime(2025, 1, 1, 9, 0), 5040),
        ]
        
        query_times = []
        for start_dt, job_minutes in test_cases:
            start_time = time.perf_counter()
            result = add_working_minutes(start_dt, job_minutes, working_intervals)
            query_time = time.perf_counter() - start_time
            query_times.append(query_time)
        
        return {
            'data_loading_time': data_loading_time,
            'interval_building_time': interval_building_time,
            'avg_query_time': sum(query_times) / len(query_times),
            'max_query_time': max(query_times),
            'total_intervals': len(working_intervals)
        }

def run_enhanced_test_suite():
    """Run the enhanced test suite with metrics."""
    print("Enhanced Working Calendar Test Suite with Metrics")
    print("="*60)
    
    # Initialize tracker
    tracker = RefactorProgressTracker('src/working_calendar.py')
    
    # Measure current state
    current_state = tracker.measure_current_state()
    
    # Print metrics
    print("\n" + "="*60)
    print("CODE SIMPLICITY METRICS")
    print("="*60)
    
    simplicity = current_state['simplicity']
    print(f"Lines of Code (total): {simplicity['total_lines']}")
    print(f"Lines of Code (code only): {simplicity['code_lines']}")
    print(f"Comment lines: {simplicity['comment_lines']}")
    print(f"Blank lines: {simplicity['blank_lines']}")
    print(f"Functions: {simplicity['functions']}")
    print(f"Classes: {simplicity['classes']}")
    print(f"Average function length: {simplicity['avg_function_length']:.1f} lines")
    print(f"Max function length: {simplicity['max_function_length']} lines")
    print(f"Average function parameters: {simplicity['avg_function_params']:.1f}")
    print(f"Max function parameters: {simplicity['max_function_params']}")
    
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    
    performance = current_state['performance']
    print(f"Data loading time: {performance['data_loading_time']*1000:.2f} ms")
    print(f"Interval building time: {performance['interval_building_time']*1000:.2f} ms")
    print(f"Average query time: {performance['avg_query_time']*1000:.3f} ms")
    print(f"Max query time: {performance['max_query_time']*1000:.3f} ms")
    print(f"Total intervals generated: {performance['total_intervals']}")
    
    print("\n" + "="*60)
    print("CORRECTNESS TESTS")
    print("="*60)
    
    # Run correctness tests
    tester = WorkingCalendarTester('data/calendar_rules.csv', 'data/calendar_exceptions.csv')
    
    # Load current implementation
    rules_df = pd.read_csv('data/calendar_rules.csv')
    exceptions_df = pd.read_csv('data/calendar_exceptions.csv')
    rules = load_calendar_rules(rules_df)
    exceptions = load_calendar_exceptions(exceptions_df)
    
    # Test cases
    test_cases = [
        ("Test 1: calA, 120 minutes", "calA", datetime(2025, 1, 6, 9, 0), 120),
        ("Test 2: calB, 120 minutes", "calB", datetime(2025, 1, 6, 9, 0), 120),
        ("Test 3: calC, 60 minutes", "calC", datetime(2025, 1, 6, 11, 50), 60),
        ("Test 4: calC, 60 minutes (deducting)", "calC", datetime(2025, 1, 1, 9, 0), 60),
        ("Test 5: calC, 5040 minutes (84 hours)", "calC", datetime(2025, 1, 1, 9, 0), 5040),
    ]
    
    all_passed = True
    
    for test_name, calendar_id, start_dt, job_minutes in test_cases:
        print(f"\n{test_name}")
        print("-" * 40)
        
        try:
            # Current implementation
            calendar_start_date = datetime(2025, 1, 1).date()
            calendar_end_date = datetime(2025, 12, 31).date()
            working_intervals = build_working_intervals(
                rules, exceptions, calendar_id, calendar_start_date, calendar_end_date
            )
            
            start_time = time.perf_counter()
            current_result = add_working_minutes(start_dt, job_minutes, working_intervals)
            current_time = time.perf_counter() - start_time
            
            # Reference implementation
            start_time = time.perf_counter()
            reference_result = tester.calculate_completion_time(start_dt, job_minutes, calendar_id)
            reference_time = time.perf_counter() - start_time
            
            print(f"Current: {current_result} ({current_time*1000:.3f} ms)")
            print(f"Reference: {reference_result} ({reference_time*1000:.3f} ms)")
            
            if current_result == reference_result:
                print("✓ PASS")
                speedup = reference_time / current_time if current_time > 0 else float('inf')
                print(f"Speedup: {speedup:.1f}x")
            else:
                print("✗ FAIL - Results differ")
                all_passed = False
                
        except Exception as e:
            print(f"✗ ERROR: {e}")
            all_passed = False
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    print(f"Timestamp: {current_state['timestamp']}")
    print(f"Code Lines: {simplicity['code_lines']}")
    print(f"Functions: {simplicity['functions']}")
    print(f"Avg Function Length: {simplicity['avg_function_length']:.1f} lines")
    print(f"Avg Query Time: {performance['avg_query_time']*1000:.3f} ms")
    print(f"Correctness: {'✓ PASS' if all_passed else '✗ FAIL'}")
    
    # Save baseline for comparison
    import json
    with open('refactor_baseline.json', 'w') as f:
        json.dump(current_state, f, indent=2)
    
    print(f"\nBaseline saved to refactor_baseline.json")
    
    return all_passed

if __name__ == '__main__':
    success = run_enhanced_test_suite()
    sys.exit(0 if success else 1)