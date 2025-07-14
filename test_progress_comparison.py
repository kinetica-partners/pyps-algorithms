import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import time
import json
from typing import Dict, Any, Optional

# Add the src directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from working_calendar import (
    load_calendar_rules, load_calendar_exceptions, build_working_intervals, add_working_minutes
)
from test_working_calendar import WorkingCalendarTester

def load_baseline() -> Optional[Dict[str, Any]]:
    """Load baseline metrics from original version."""
    try:
        with open('original_baseline.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def compare_metrics(current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Compare current metrics with baseline."""
    comparison = {}
    
    # Simplicity comparisons
    current_simplicity = current['simplicity']
    baseline_simplicity = baseline['simplicity']
    
    comparison['simplicity'] = {
        'code_lines': {
            'current': current_simplicity['code_lines'],
            'baseline': baseline_simplicity['code_lines'],
            'change': current_simplicity['code_lines'] - baseline_simplicity['code_lines'],
            'pct_change': ((current_simplicity['code_lines'] - baseline_simplicity['code_lines']) / baseline_simplicity['code_lines'] * 100) if baseline_simplicity['code_lines'] > 0 else 0
        },
        'functions': {
            'current': current_simplicity['functions'],
            'baseline': baseline_simplicity['functions'],
            'change': current_simplicity['functions'] - baseline_simplicity['functions'],
            'pct_change': ((current_simplicity['functions'] - baseline_simplicity['functions']) / baseline_simplicity['functions'] * 100) if baseline_simplicity['functions'] > 0 else 0
        },
        'avg_function_length': {
            'current': current_simplicity['avg_function_length'],
            'baseline': baseline_simplicity['avg_function_length'],
            'change': current_simplicity['avg_function_length'] - baseline_simplicity['avg_function_length'],
            'pct_change': ((current_simplicity['avg_function_length'] - baseline_simplicity['avg_function_length']) / baseline_simplicity['avg_function_length'] * 100) if baseline_simplicity['avg_function_length'] > 0 else 0
        },
        'max_function_length': {
            'current': current_simplicity['max_function_length'],
            'baseline': baseline_simplicity['max_function_length'],
            'change': current_simplicity['max_function_length'] - baseline_simplicity['max_function_length'],
            'pct_change': ((current_simplicity['max_function_length'] - baseline_simplicity['max_function_length']) / baseline_simplicity['max_function_length'] * 100) if baseline_simplicity['max_function_length'] > 0 else 0
        }
    }
    
    # Performance comparisons
    current_performance = current['performance']
    baseline_performance = baseline['performance']
    
    comparison['performance'] = {
        'avg_query_time': {
            'current': current_performance['avg_query_time'],
            'baseline': baseline_performance['avg_query_time'],
            'change': current_performance['avg_query_time'] - baseline_performance['avg_query_time'],
            'pct_change': ((current_performance['avg_query_time'] - baseline_performance['avg_query_time']) / baseline_performance['avg_query_time'] * 100) if baseline_performance['avg_query_time'] > 0 else 0
        },
        'interval_building_time': {
            'current': current_performance['interval_building_time'],
            'baseline': baseline_performance['interval_building_time'],
            'change': current_performance['interval_building_time'] - baseline_performance['interval_building_time'],
            'pct_change': ((current_performance['interval_building_time'] - baseline_performance['interval_building_time']) / baseline_performance['interval_building_time'] * 100) if baseline_performance['interval_building_time'] > 0 else 0
        }
    }
    
    return comparison

def run_progress_comparison():
    """Run comparison test showing progress from baseline."""
    print("Working Calendar Refactor Progress Comparison")
    print("="*60)
    
    # Load baseline
    baseline = load_baseline()
    if baseline is None:
        print("No baseline found! Run test_refactor_progress.py first.")
        return False
    
    # Get current metrics using the same logic as the progress tracker
    from test_refactor_progress import RefactorProgressTracker
    tracker = RefactorProgressTracker('src/working_calendar.py')
    current_state = tracker.measure_current_state()
    
    # Compare metrics
    comparison = compare_metrics(current_state, baseline)
    
    print(f"Baseline from: {baseline['timestamp']}")
    print(f"Current run:   {current_state['timestamp']}")
    
    print("\n" + "="*60)
    print("SIMPLICITY PROGRESS")
    print("="*60)
    
    # Code lines
    cl = comparison['simplicity']['code_lines']
    print(f"Code Lines:     {cl['baseline']} → {cl['current']} ({cl['change']:+d}, {cl['pct_change']:+.1f}%)")
    
    # Functions
    fc = comparison['simplicity']['functions']
    print(f"Functions:      {fc['baseline']} → {fc['current']} ({fc['change']:+d}, {fc['pct_change']:+.1f}%)")
    
    # Average function length
    afl = comparison['simplicity']['avg_function_length']
    print(f"Avg Func Len:   {afl['baseline']:.1f} → {afl['current']:.1f} ({afl['change']:+.1f}, {afl['pct_change']:+.1f}%)")
    
    # Max function length
    mfl = comparison['simplicity']['max_function_length']
    print(f"Max Func Len:   {mfl['baseline']} → {mfl['current']} ({mfl['change']:+d}, {mfl['pct_change']:+.1f}%)")
    
    print("\n" + "="*60)
    print("PERFORMANCE PROGRESS")
    print("="*60)
    
    # Average query time
    aqt = comparison['performance']['avg_query_time']
    print(f"Avg Query Time: {aqt['baseline']*1000:.3f} ms → {aqt['current']*1000:.3f} ms ({aqt['change']*1000:+.3f} ms, {aqt['pct_change']:+.1f}%)")
    
    # Interval building time
    ibt = comparison['performance']['interval_building_time']
    print(f"Interval Build: {ibt['baseline']*1000:.2f} ms → {ibt['current']*1000:.2f} ms ({ibt['change']*1000:+.2f} ms, {ibt['pct_change']:+.1f}%)")
    
    print("\n" + "="*60)
    print("CORRECTNESS VERIFICATION")
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
        print(f"{test_name}: ", end="")
        
        try:
            # Current implementation
            calendar_start_date = datetime(2025, 1, 1).date()
            calendar_end_date = datetime(2025, 12, 31).date()
            working_intervals = build_working_intervals(
                rules, exceptions, calendar_id, calendar_start_date, calendar_end_date
            )
            
            current_result = add_working_minutes(start_dt, job_minutes, working_intervals)
            
            # Reference implementation (for correctness only)
            reference_result = tester.calculate_completion_time(start_dt, job_minutes, calendar_id)
            
            if current_result == reference_result:
                print("✓ PASS")
            else:
                print("✗ FAIL")
                all_passed = False
                
        except Exception as e:
            print(f"✗ ERROR: {e}")
            all_passed = False
    
    print(f"\n{'='*60}")
    print("REFACTOR ASSESSMENT")
    print(f"{'='*60}")
    
    # Overall assessment
    simplicity_improved = (afl['change'] < 0 or mfl['change'] < 0)  # Lower is better
    performance_maintained = abs(aqt['pct_change']) < 10  # Within 10% is maintained
    correctness_maintained = all_passed
    
    print(f"Simplicity:   {'✓ IMPROVED' if simplicity_improved else '○ UNCHANGED'}")
    print(f"Performance:  {'✓ MAINTAINED' if performance_maintained else '⚠ CHANGED'}")
    print(f"Correctness:  {'✓ MAINTAINED' if correctness_maintained else '✗ FAILED'}")
    
    if simplicity_improved and performance_maintained and correctness_maintained:
        print("\n🎉 SUCCESSFUL REFACTOR!")
    elif correctness_maintained:
        print("\n⚠ REFACTOR COMPLETE (with trade-offs)")
    else:
        print("\n❌ REFACTOR FAILED")
    
    # Save current state as new baseline
    with open('refactor_baseline.json', 'w') as f:
        json.dump(current_state, f, indent=2)
    
    return all_passed

if __name__ == '__main__':
    success = run_progress_comparison()
    sys.exit(0 if success else 1)