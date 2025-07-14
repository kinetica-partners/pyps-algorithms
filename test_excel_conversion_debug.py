import pandas as pd
from datetime import datetime
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from working_calendar import datetime_to_excel

def test_excel_conversion():
    """Debug the Excel date conversion."""
    
    print("=== Debugging Excel Date Conversion ===")
    
    # Test our function
    test_dt = datetime(2025, 1, 1, 10, 0, 0)
    print(f"Input datetime: {test_dt}")
    
    result = datetime_to_excel(test_dt)
    print(f"Our function result: {result}")
    
    # What Excel should expect for 2025-01-01 10:00:00
    # Excel serial date for 2025-01-01 is 45658
    # 10:00:00 is 10/24 = 0.41666... days
    expected_excel = 45658 + (10/24)
    print(f"Expected Excel value: {expected_excel}")
    
    # Test with pandas (which should be correct)
    excel_epoch = datetime(1899, 12, 30)
    delta = test_dt - excel_epoch
    manual_calc = delta.days + (delta.seconds / 86400)
    print(f"Manual calculation: {manual_calc}")
    
    # Check if the issue is with string parsing
    test_string = "2025-01-01 00:00:00"
    parsed_dt = pd.to_datetime(test_string)
    print(f"Pandas parsed '{test_string}' as: {parsed_dt} (type: {type(parsed_dt)})")
    if hasattr(parsed_dt, 'to_pydatetime'):
        py_dt = parsed_dt.to_pydatetime()
        print(f"Converted to Python datetime: {py_dt} (type: {type(py_dt)})")
        
        # Test our conversion on the parsed datetime
        converted = datetime_to_excel(py_dt)
        print(f"Our function on pandas-parsed datetime: {converted}")

if __name__ == "__main__":
    test_excel_conversion()