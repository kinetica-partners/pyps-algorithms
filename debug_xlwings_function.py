import pandas as pd
from datetime import datetime, date, timedelta
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import our function (but we'll test it without the xlwings decorator)
from working_calendar import (
    load_calendar_rules, 
    load_calendar_exceptions,
    build_working_intervals,
    add_working_minutes
)

def debug_xlwings_function():
    """Debug the xlwings function logic with detailed logging."""
    
    print("=== Debug xlwings function ===")
    print()
    
    # Test parameters
    start_datetime = datetime(2025, 1, 1, 0, 0)  # Start at midnight
    jobtime = 60.0  # 60 minutes
    calendar_id = "calC"
    
    print(f"Input parameters:")
    print(f"  start_datetime: {start_datetime}")
    print(f"  jobtime: {jobtime} minutes")
    print(f"  calendar_id: {calendar_id}")
    print()
    
    try:
        # Load CSV data
        print("Loading CSV data...")
        rules_df = pd.read_csv('data/calendar_rules.csv')
        exceptions_df = pd.read_csv('data/calendar_exceptions.csv')
        
        print(f"Rules DataFrame shape: {rules_df.shape}")
        print(f"Exceptions DataFrame shape: {exceptions_df.shape}")
        print()
        
        # Convert to 2D arrays like Excel would pass
        print("Converting to 2D arrays (like Excel would pass)...")
        
        # Rules data
        rules_headers = rules_df.columns.tolist()
        rules_data = rules_df.values.tolist()
        calendar_rules_data = [rules_headers] + rules_data
        
        # Exceptions data
        exceptions_headers = exceptions_df.columns.tolist()
        exceptions_data = exceptions_df.values.tolist()
        calendar_exceptions_data = [exceptions_headers] + exceptions_data
        
        print(f"Rules data shape: {len(calendar_rules_data)} rows x {len(calendar_rules_data[0])} cols")
        print(f"Exceptions data shape: {len(calendar_exceptions_data)} rows x {len(calendar_exceptions_data[0])} cols")
        print()
        
        # Show sample data
        print("Sample rules data:")
        for i, row in enumerate(calendar_rules_data[:3]):
            print(f"  Row {i}: {row}")
        print()
        
        print("Sample exceptions data:")
        for i, row in enumerate(calendar_exceptions_data[:3]):
            print(f"  Row {i}: {row}")
        print()
        
        # Now simulate the xlwings function logic
        print("=== Simulating xlwings function logic ===")
        print()
        
        # Step 1: Convert to DataFrames
        print("Step 1: Converting 2D arrays to DataFrames...")
        
        if not calendar_rules_data or len(calendar_rules_data) < 2:
            print("ERROR: Calendar rules data is empty or missing headers")
            return
        
        if not calendar_exceptions_data or len(calendar_exceptions_data) < 1:
            print("ERROR: Calendar exceptions data is empty")
            return
        
        # Create DataFrames from 2D arrays
        rules_headers = calendar_rules_data[0]
        rules_data = calendar_rules_data[1:]
        new_rules_df = pd.DataFrame(rules_data, columns=rules_headers)
        
        exceptions_headers = calendar_exceptions_data[0]
        exceptions_data = calendar_exceptions_data[1:] if len(calendar_exceptions_data) > 1 else []
        new_exceptions_df = pd.DataFrame(exceptions_data, columns=exceptions_headers)
        
        print(f"Recreated rules DataFrame shape: {new_rules_df.shape}")
        print(f"Recreated exceptions DataFrame shape: {new_exceptions_df.shape}")
        print()
        
        # Step 2: Load calendar data
        print("Step 2: Loading calendar data...")
        rules = load_calendar_rules(new_rules_df)
        exceptions = load_calendar_exceptions(new_exceptions_df)
        
        print(f"Rules loaded for calendar IDs: {list(rules.keys())}")
        print(f"Exceptions loaded for calendar IDs: {list(exceptions.keys())}")
        print()
        
        # Step 3: Check calendar ID
        print("Step 3: Checking calendar ID...")
        if calendar_id not in rules:
            print(f"Warning: Calendar ID '{calendar_id}' not found in rules")
            if not rules:
                print("ERROR: No calendar rules found in data")
                return
            calendar_id = next(iter(rules.keys()))
            print(f"Using first available calendar ID: {calendar_id}")
        
        print(f"Using calendar ID: {calendar_id}")
        print()
        
        # Step 4: Build working intervals
        print("Step 4: Building working intervals...")
        start_date = start_datetime.date()
        minutes_to_add = round(float(jobtime))
        
        # Add buffer based on job size
        buffer_days = max(7, int(minutes_to_add / (8 * 60)) + 1)
        end_date = start_date + timedelta(days=buffer_days)
        
        print(f"Start date: {start_date}")
        print(f"End date: {end_date}")
        print(f"Minutes to add: {minutes_to_add}")
        print()
        
        working_intervals = build_working_intervals(
            rules, exceptions, calendar_id, start_date, end_date
        )
        
        print(f"Built {len(working_intervals)} working intervals")
        
        # Show first few intervals
        print("First 5 working intervals:")
        for i, (start, end, cumu) in enumerate(working_intervals[:5]):
            print(f"  {i+1}: {start} to {end} (cumulative: {cumu} mins)")
        print()
        
        # Step 5: Calculate completion time
        print("Step 5: Calculating completion time...")
        completion_dt = add_working_minutes(start_datetime, minutes_to_add, working_intervals)
        
        if completion_dt is None:
            print(f"ERROR: Cannot complete {minutes_to_add} minutes of work within the calculated time range")
            return
        
        print(f"SUCCESS: Completion time calculated: {completion_dt}")
        print()
        
        # Summary
        print("=== SUMMARY ===")
        print(f"Start:      {start_datetime}")
        print(f"Job time:   {minutes_to_add} minutes")
        print(f"Calendar:   {calendar_id}")
        print(f"Completion: {completion_dt}")
        print()
        
        return completion_dt
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    debug_xlwings_function()