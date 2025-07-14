import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from working_calendar import (
    load_calendar_rules, load_calendar_exceptions, build_working_intervals, add_working_minutes,
    get_working_periods_for_day
)
from test_working_calendar import WorkingCalendarTester

def debug_large_test():
    """Debug the large test case to understand the discrepancy."""
    print("Debugging Large Test Case (84 hours)")
    print("="*50)
    
    # Load data
    rules_df = pd.read_csv('data/calendar_rules.csv')
    exceptions_df = pd.read_csv('data/calendar_exceptions.csv')
    
    # Initialize current implementation
    rules = load_calendar_rules(rules_df)
    exceptions = load_calendar_exceptions(exceptions_df)
    
    # Initialize reference implementation
    tester = WorkingCalendarTester('data/calendar_rules.csv', 'data/calendar_exceptions.csv')
    
    # Test parameters
    calendar_id = 'calC'
    start_dt = datetime(2025, 1, 1, 9, 0)
    job_minutes = 5040
    
    print(f"Calendar: {calendar_id}")
    print(f"Start: {start_dt}")
    print(f"Job Duration: {job_minutes} minutes")
    
    # Current implementation
    calendar_start_date = datetime(2025, 1, 1).date()
    calendar_end_date = datetime(2025, 12, 31).date()
    working_intervals = build_working_intervals(
        rules, exceptions, calendar_id, calendar_start_date, calendar_end_date
    )
    current_result = add_working_minutes(start_dt, job_minutes, working_intervals)
    
    # Reference implementation
    reference_result = tester.calculate_completion_time(start_dt, job_minutes, calendar_id)
    
    print(f"\nCurrent Implementation: {current_result}")
    print(f"Reference Implementation: {reference_result}")
    
    if current_result and reference_result:
        diff = (current_result - reference_result).total_seconds()
        print(f"Difference: {diff} seconds ({diff/60} minutes)")
    
    # Let's examine the working intervals for the first few days
    print(f"\nExamining working intervals for first few days:")
    for i, (start, end, cumulative) in enumerate(working_intervals[:20]):
        print(f"  {i:2d}: {start} to {end} (cumulative: {cumulative} minutes)")
    
    # Let's check the working periods calculation for the first few days
    print(f"\nExamining working periods for first few days:")
    for day_offset in range(8):
        current_date = (start_dt + timedelta(days=day_offset)).date()
        periods = get_working_periods_for_day(current_date, calendar_id, rules, exceptions)
        
        # Calculate total minutes for this day
        total_minutes = 0
        for start_time, end_time in periods:
            start_dt_day = datetime.combine(current_date, start_time)
            end_dt_day = datetime.combine(current_date, end_time)
            total_minutes += int((end_dt_day - start_dt_day).total_seconds() // 60)
        
        print(f"  {current_date}: {periods} -> {total_minutes} minutes")
        
        # Check what the reference implementation thinks
        ref_minutes = 0
        for minute in range(0, 24*60):
            test_dt = datetime.combine(current_date, datetime.min.time()) + timedelta(minutes=minute)
            if tester._is_working_minute(test_dt, calendar_id):
                ref_minutes += 1
        
        print(f"    Reference: {ref_minutes} minutes")
        
        if total_minutes != ref_minutes:
            print(f"    *** MISMATCH: Current={total_minutes}, Reference={ref_minutes}")

if __name__ == '__main__':
    debug_large_test()