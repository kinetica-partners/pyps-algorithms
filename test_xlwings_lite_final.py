import pandas as pd
from datetime import datetime
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from working_calendar import calculate_working_completion_time, test_working_calendar

def test_xlwings_lite_final():
    """Test the final xlwings lite compatible function with 2D array arguments."""
    
    print("=== Testing Final xlwings Lite Compatible Function ===")
    
    # Test data as 2D arrays (as Excel ranges would be passed)
    calendar_rules_data = [
        ['id', 'calendar_id', 'weekday', 'start_time', 'end_time'],
        [1, 'calC', 'Mon', '09:00', '21:00'],
        [2, 'calC', 'Tue', '09:00', '21:00'],
        [3, 'calC', 'Wed', '09:00', '21:00'],
        [4, 'calC', 'Thu', '09:00', '21:00'],
        [5, 'calC', 'Fri', '09:00', '21:00']
    ]
    
    calendar_exceptions_data = [
        ['id', 'calendar_id', 'date', 'start_time', 'end_time', 'is_working'],
        [1, 'calC', '2025-01-01', '09:00', '09:10', False]
    ]
    
    # Test parameters (as Excel would pass them)
    start_datetime = datetime(2025, 1, 1, 0, 0)
    jobtime = 60.0
    calendar_id = "calC"
    
    print(f"Input parameters:")
    print(f"  start_datetime: {start_datetime} (type: {type(start_datetime)})")
    print(f"  jobtime: {jobtime} (type: {type(jobtime)})")
    print(f"  calendar_id: {calendar_id} (type: {type(calendar_id)})")
    print(f"  rules data (2D array): {len(calendar_rules_data)} rows")
    print(f"  exceptions data (2D array): {len(calendar_exceptions_data)} rows")
    print()
    
    try:
        # Call the function with 2D array arguments
        result = calculate_working_completion_time(
            start_datetime,
            jobtime,
            calendar_rules_data,
            calendar_exceptions_data,
            calendar_id
        )
        
        print(f"Function result: {result}")
        print(f"Result type: {type(result)}")
        
        if isinstance(result, float):
            print(f"✅ SUCCESS: Function returned Excel date value: {result}")
            
            # Convert back to datetime for verification
            excel_epoch = datetime(1899, 12, 30)
            result_dt = excel_epoch + pd.Timedelta(days=result)
            print(f"✅ Converted back to datetime: {result_dt}")
            
            # Expected datetime result
            expected_dt = datetime(2025, 1, 1, 10, 10, 0)
            print(f"✅ Expected datetime: {expected_dt}")
            
            # Check if they match (within 1 second tolerance)
            time_diff = abs((result_dt - expected_dt).total_seconds())
            if time_diff < 1:
                print("✅ PERFECT: Excel date conversion and calculation working correctly!")
                return True
            else:
                print(f"⚠️  DIFFERENCE: {time_diff} seconds difference")
                return False
                
        elif isinstance(result, str) and result.startswith("Error:"):
            print(f"❌ FUNCTION ERROR: {result}")
            return False
        else:
            print(f"❌ UNEXPECTED RESULT TYPE: {result}")
            return False
            
    except Exception as e:
        print(f"❌ PYTHON ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_function():
    """Test the simple test function."""
    print("\n=== Testing Simple Test Function ===")
    
    try:
        result = test_working_calendar()
        print(f"test_working_calendar() result: {result}")
        print(f"Result type: {type(result)}")
        
        if isinstance(result, float):
            print(f"✅ SUCCESS: test_working_calendar returned Excel date value: {result}")
            return True
        else:
            print(f"❌ UNEXPECTED: test_working_calendar returned: {result}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR in test_working_calendar: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_with_defaults():
    """Test the function with default data."""
    print("\n=== Testing Function with Default Data ===")
    
    try:
        # Test with minimal arguments - should use defaults
        start_datetime = datetime(2025, 1, 1, 9, 0)  # Start at 9 AM
        jobtime = 60.0
        
        result = calculate_working_completion_time(
            start_datetime,
            jobtime
        )
        
        print(f"Function result with defaults: {result}")
        print(f"Result type: {type(result)}")
        
        if isinstance(result, float):
            print(f"✅ SUCCESS: Function with defaults returned Excel date value: {result}")
            return True
        else:
            print(f"❌ ISSUE: Function with defaults returned: {result}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR with defaults: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_simple_function()
    success2 = test_with_defaults()
    success3 = test_xlwings_lite_final()
    
    if success1 and success2 and success3:
        print("\n🎉 ALL TESTS PASSED! The xlwings lite function is ready for Excel integration.")
        print("📋 Next steps:")
        print("   1. The function no longer uses @arg decorators")
        print("   2. All data is passed as function arguments")
        print("   3. The function handles 2D arrays internally")
        print("   4. Returns Excel-compatible date values")
        print("   5. Should work correctly when called from Excel")
    else:
        print("\n❌ SOME TESTS FAILED. There are still issues to resolve.")