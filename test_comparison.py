import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from working_calendar import (
    load_calendar_rules, load_calendar_exceptions, build_working_intervals, add_working_minutes
)
from test_working_calendar import WorkingCalendarTester

def test_comparison():
    """Compare the current implementation with the reference implementation."""
    print("Working Calendar Implementation Comparison")
    print("="*60)
    
    # Load data
    rules_df = pd.read_csv('data/calendar_rules.csv')
    exceptions_df = pd.read_csv('data/calendar_exceptions.csv')
    
    # Initialize current implementation
    rules = load_calendar_rules(rules_df)
    exceptions = load_calendar_exceptions(exceptions_df)
    
    # Initialize reference implementation
    tester = WorkingCalendarTester('data/calendar_rules.csv', 'data/calendar_exceptions.csv')
    
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
        print(f"Calendar: {calendar_id}")
        print(f"Start: {start_dt}")
        print(f"Job Duration: {job_minutes} minutes")
        
        try:
            # Current implementation
            calendar_start_date = datetime(2025, 1, 1).date()
            calendar_end_date = datetime(2025, 12, 31).date()
            working_intervals = build_working_intervals(
                rules, exceptions, calendar_id, calendar_start_date, calendar_end_date
            )
            current_result = add_working_minutes(start_dt, job_minutes, working_intervals)
            
            # Reference implementation
            reference_result = tester.calculate_completion_time(start_dt, job_minutes, calendar_id)
            
            print(f"Current Implementation: {current_result}")
            print(f"Reference Implementation: {reference_result}")
            
            if current_result == reference_result:
                print("✓ PASS - Results match")
            else:
                print("✗ FAIL - Results differ")
                if current_result and reference_result:
                    diff = abs((current_result - reference_result).total_seconds())
                    print(f"  Difference: {diff} seconds")
                all_passed = False
                
        except Exception as e:
            print(f"✗ ERROR: {e}")
            all_passed = False
    
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    if all_passed:
        print("✓ All tests passed - Current implementation matches reference")
        return True
    else:
        print("✗ Some tests failed - Current implementation differs from reference")
        return False

if __name__ == '__main__':
    success = test_comparison()
    sys.exit(0 if success else 1)