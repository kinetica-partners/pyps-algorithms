import pandas as pd
from datetime import datetime
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from working_calendar import calculate_working_completion_time

def test_string_datetime_inputs():
    """Test the function with string datetime inputs as Excel sends them."""
    
    print("=== Testing String Datetime Inputs (Excel Format) ===")
    
    # Test cases mimicking what Excel sends
    test_cases = [
        {
            "name": "String datetime like Excel text",
            "start_datetime": "2025-01-01 00:00:00",
            "jobtime": 60,
            "expected_result": "Excel date value"
        },
        {
            "name": "Date-only string",
            "start_datetime": "2025-01-01",
            "jobtime": 60,
            "expected_result": "Excel date value"
        },
        {
            "name": "Pandas Timestamp",
            "start_datetime": pd.Timestamp("2025-01-01 00:00:00"),
            "jobtime": 60,
            "expected_result": "Excel date value"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        print(f"Input: start_datetime='{test_case['start_datetime']}' (type: {type(test_case['start_datetime'])}), jobtime={test_case['jobtime']}")
        
        try:
            result = calculate_working_completion_time(
                test_case['start_datetime'],
                test_case['jobtime']
            )
            
            print(f"Result: {result}")
            print(f"Result type: {type(result)}")
            
            if isinstance(result, float):
                print(f"✅ SUCCESS: Returned Excel date value {result}")
                
                # Convert back to verify
                excel_epoch = datetime(1899, 12, 30)
                result_dt = excel_epoch + pd.Timedelta(days=result)
                print(f"   Converts to datetime: {result_dt}")
                
            elif isinstance(result, str) and result.startswith("Error:"):
                print(f"❌ ERROR: {result}")
            else:
                print(f"⚠️  UNEXPECTED: {result}")
                
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_string_datetime_inputs()