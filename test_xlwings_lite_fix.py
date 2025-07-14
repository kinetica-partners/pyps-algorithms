import pandas as pd
from datetime import datetime
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from working_calendar import calculate_working_completion_time

def test_xlwings_lite_function():
    """Test the function using the xlwings Lite approach with DataFrames."""
    
    print("=== Testing xlwings Lite Compatible Function ===")
    
    # Create test data as DataFrames (as xlwings Lite would provide)
    rules_data = {
        'id': [1, 2, 3, 4, 5],
        'calendar_id': ['calC', 'calC', 'calC', 'calC', 'calC'],
        'weekday': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
        'start_time': ['09:00', '09:00', '09:00', '09:00', '09:00'],
        'end_time': ['21:00', '21:00', '21:00', '21:00', '21:00']
    }
    calendar_rules_data = pd.DataFrame(rules_data)
    
    exceptions_data = {
        'id': [1],
        'calendar_id': ['calC'],
        'date': ['2025-01-01'],
        'start_time': ['09:00'],
        'end_time': ['09:10'],
        'is_working': [False]
    }
    calendar_exceptions_data = pd.DataFrame(exceptions_data)
    
    # Test parameters (as xlwings Lite would convert them)
    start_datetime = datetime(2025, 1, 1, 0, 0)  # Excel datetime converted to Python datetime
    jobtime = 60.0  # Excel number converted to Python float
    calendar_id = "calC"  # Excel string
    
    print(f"Input parameters:")
    print(f"  start_datetime: {start_datetime} (type: {type(start_datetime)})")
    print(f"  jobtime: {jobtime} (type: {type(jobtime)})")
    print(f"  calendar_id: {calendar_id} (type: {type(calendar_id)})")
    print(f"  rules DataFrame shape: {calendar_rules_data.shape}")
    print(f"  exceptions DataFrame shape: {calendar_exceptions_data.shape}")
    print()
    
    print("Calendar rules data:")
    print(calendar_rules_data)
    print()
    
    print("Calendar exceptions data:")
    print(calendar_exceptions_data)
    print()
    
    try:
        # Call the function
        result = calculate_working_completion_time(
            start_datetime,
            jobtime,
            calendar_rules_data,
            calendar_exceptions_data,
            calendar_id
        )
        
        print(f"Function result: {result}")
        print(f"Result type: {type(result)}")
        
        if isinstance(result, datetime):
            print(f"✓ SUCCESS: Function returned datetime: {result}")
            print(f"✓ Expected: 2025-01-01 10:10:00 (midnight + 60 minutes working time)")
            
            # Verify the result
            expected = datetime(2025, 1, 1, 10, 10)
            if result == expected:
                print("✓ PERFECT: Result matches expected value!")
            else:
                print(f"⚠️  DIFFERENCE: Expected {expected}, got {result}")
                
        else:
            print(f"❌ ISSUE: Function returned: {result}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_xlwings_lite_function()
    if success:
        print("\n🎉 All tests passed! The xlwings Lite function should now work correctly.")
    else:
        print("\n❌ Tests failed. There are still issues to resolve.")