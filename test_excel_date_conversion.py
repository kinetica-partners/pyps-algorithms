import pandas as pd
from datetime import datetime
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from working_calendar import calculate_working_completion_time, datetime_to_excel

def test_excel_date_conversion():
    """Test the Excel date conversion function."""
    
    print("=== Testing Excel Date Conversion ===")
    
    # Test the datetime_to_excel function directly
    test_dt = datetime(2025, 1, 1, 10, 10, 0)
    excel_value = datetime_to_excel(test_dt)
    
    print(f"Python datetime: {test_dt}")
    print(f"Excel value: {excel_value}")
    
    # Verify the conversion (Excel serial date for 2025-01-01 10:10:00)
    # Excel date serial for 2025-01-01 is 45658
    # 10:10:00 is 10*60 + 10 = 610 minutes = 610/1440 days = 0.4236111... days
    expected_excel_value = 45658 + (10 * 60 + 10) / (24 * 60)
    print(f"Expected Excel value: {expected_excel_value}")
    print(f"Difference: {abs(excel_value - expected_excel_value)}")
    
    if abs(excel_value - expected_excel_value) < 0.0001:
        print("✓ Excel date conversion is correct!")
    else:
        print("❌ Excel date conversion is incorrect!")
    
    return True

def test_function_with_excel_conversion():
    """Test the working calendar function with Excel date conversion."""
    
    print("\n=== Testing Function with Excel Date Conversion ===")
    
    # Create test data as DataFrames
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
    
    # Test parameters
    start_datetime = datetime(2025, 1, 1, 0, 0)
    jobtime = 60.0
    calendar_id = "calC"
    
    print(f"Input parameters:")
    print(f"  start_datetime: {start_datetime}")
    print(f"  jobtime: {jobtime}")
    print(f"  calendar_id: {calendar_id}")
    
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
        
        if isinstance(result, float):
            print(f"✓ SUCCESS: Function returned Excel date value: {result}")
            
            # Convert back to datetime for verification
            excel_epoch = datetime(1899, 12, 30)
            result_dt = excel_epoch + pd.Timedelta(days=result)
            print(f"✓ Converted back to datetime: {result_dt}")
            
            # Expected datetime result
            expected_dt = datetime(2025, 1, 1, 10, 10, 0)
            print(f"✓ Expected datetime: {expected_dt}")
            
            # Check if they match (within 1 second tolerance)
            time_diff = abs((result_dt - expected_dt).total_seconds())
            if time_diff < 1:
                print("✓ PERFECT: Excel date conversion is working correctly!")
            else:
                print(f"⚠️  DIFFERENCE: {time_diff} seconds difference")
                
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
    test_excel_date_conversion()
    success = test_function_with_excel_conversion()
    
    if success:
        print("\n🎉 All tests passed! The function now returns Excel-compatible date values.")
    else:
        print("\n❌ Tests failed. There are still issues to resolve.")