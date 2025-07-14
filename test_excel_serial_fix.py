import pandas as pd
from datetime import datetime
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from working_calendar import calculate_working_completion_time

def test_excel_serial_fix():
    """Test the fix for Excel serial number handling."""
    
    print("=== Testing Excel Serial Number Fix ===")
    
    test_cases = [
        {
            "name": "Excel serial number (what DATE(2025,1,1) sends)",
            "start_datetime": 45658,  # Excel serial for 2025-01-01
            "jobtime": 60,
            "expected_date": "2025-01-01 10:00:00"
        },
        {
            "name": "Excel serial number with time (10:00 AM)",
            "start_datetime": 45658.416666666664,  # Excel serial for 2025-01-01 10:00:00
            "jobtime": 60,
            "expected_date": "2025-01-01 11:00:00"
        },
        {
            "name": "String datetime (still works)",
            "start_datetime": "2025-01-01 00:00:00",
            "jobtime": 60,
            "expected_date": "2025-01-01 10:00:00"
        },
        {
            "name": "Unix timestamp (small number)",
            "start_datetime": 1735689600,  # Unix timestamp for 2025-01-01 00:00:00 UTC
            "jobtime": 60,
            "expected_date": "2025-01-01 10:00:00"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        print(f"Input: start_datetime={test_case['start_datetime']} (type: {type(test_case['start_datetime'])}), jobtime={test_case['jobtime']}")
        print(f"Expected result date: {test_case['expected_date']}")
        
        try:
            result = calculate_working_completion_time(
                test_case['start_datetime'],
                test_case['jobtime']
            )
            
            print(f"Result: {result}")
            print(f"Result type: {type(result)}")
            
            if isinstance(result, float):
                # Convert back to verify
                excel_epoch = datetime(1899, 12, 30)
                result_dt = excel_epoch + pd.Timedelta(days=result)
                print(f"✅ SUCCESS: Excel value {result}")
                print(f"   Converts to datetime: {result_dt}")
                
                # Check if close to expected
                expected_dt = pd.to_datetime(test_case['expected_date'])
                time_diff = abs((result_dt - expected_dt).total_seconds())
                if time_diff < 60:  # Within 1 minute
                    print(f"✅ MATCHES EXPECTED: Within {time_diff} seconds of expected time")
                else:
                    print(f"⚠️  TIME DIFFERENCE: {time_diff} seconds from expected")
                
            elif isinstance(result, str) and result.startswith("Error:"):
                print(f"❌ ERROR: {result}")
            else:
                print(f"⚠️  UNEXPECTED: {result}")
                
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_excel_serial_fix()