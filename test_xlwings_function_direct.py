import pandas as pd
from datetime import datetime
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the xlwings function directly
from working_calendar import calculate_working_completion_time

def test_xlwings_function_direct():
    """Test the xlwings function directly as Excel would call it."""
    
    print("=== Testing xlwings function directly ===")
    print()
    
    # Load CSV data and convert to 2D arrays (like Excel would pass)
    rules_df = pd.read_csv('data/calendar_rules.csv')
    exceptions_df = pd.read_csv('data/calendar_exceptions.csv')
    
    # Convert to 2D arrays
    rules_headers = rules_df.columns.tolist()
    rules_data = rules_df.values.tolist()
    calendar_rules_data = [rules_headers] + rules_data
    
    exceptions_headers = exceptions_df.columns.tolist()
    exceptions_data = exceptions_df.values.tolist()
    calendar_exceptions_data = [exceptions_headers] + exceptions_data
    
    # Test parameters - exactly as Excel would call it
    start_datetime = datetime(2025, 1, 1, 0, 0)  # Excel datetime
    jobtime = 60.0  # Excel number
    calendar_id = "calC"  # Excel string
    
    print(f"Calling function with:")
    print(f"  start_datetime: {start_datetime}")
    print(f"  jobtime: {jobtime}")
    print(f"  calendar_rules_data: {len(calendar_rules_data)} rows")
    print(f"  calendar_exceptions_data: {len(calendar_exceptions_data)} rows")
    print(f"  calendar_id: {calendar_id}")
    print()
    
    # Call the function directly (without the @func decorator)
    try:
        # Remove the @func decorator temporarily for testing
        result = calculate_working_completion_time(
            start_datetime,
            jobtime,
            calendar_rules_data,
            calendar_exceptions_data,
            calendar_id
        )
        
        print(f"Function returned: {result}")
        print(f"Return type: {type(result)}")
        
        if isinstance(result, datetime):
            print(f"SUCCESS: Function returned datetime: {result}")
        else:
            print(f"ISSUE: Function returned non-datetime: {result}")
            
    except Exception as e:
        print(f"ERROR: Function failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_xlwings_function_direct()