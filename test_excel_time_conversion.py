#!/usr/bin/env python3
"""Test Excel time serial number conversion in working calendar."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from working_calendar import excel_time_to_string, parse_time_string, calculate_working_completion_time
from datetime import datetime, time
import pandas as pd

def test_excel_time_conversion():
    """Test Excel time serial number conversion."""
    print("=== Testing Excel Time Serial Number Conversion ===")
    
    # Test excel_time_to_string function
    test_cases = [
        (0.375, "09:00"),     # 9:00 AM = 9/24 = 0.375
        (0.5, "12:00"),       # 12:00 PM = 12/24 = 0.5
        (0.708333, "17:00"),  # 5:00 PM = 17/24 = 0.708333
        (0.75, "18:00"),      # 6:00 PM = 18/24 = 0.75
        (0.0, "00:00"),       # Midnight = 0/24 = 0.0
        ("09:00", "09:00"),   # String should pass through
    ]
    
    print("\nTesting excel_time_to_string():")
    for excel_time, expected in test_cases:
        result = excel_time_to_string(excel_time)
        status = "✓" if result == expected else "✗"
        print(f"{status} {excel_time} -> {result} (expected: {expected})")
    
    # Test parse_time_string function
    print("\nTesting parse_time_string():")
    for excel_time, expected_str in test_cases:
        try:
            result = parse_time_string(excel_time)
            expected_time = time.fromisoformat(expected_str)
            status = "✓" if result == expected_time else "✗"
            print(f"{status} {excel_time} -> {result} (expected: {expected_time})")
        except Exception as e:
            print(f"✗ {excel_time} -> ERROR: {e}")

def test_calendar_with_excel_times():
    """Test calendar function with Excel time serial numbers."""
    print("\n=== Testing Calendar with Excel Time Serial Numbers ===")
    
    # Create test calendar data with Excel time serial numbers
    calendar_rules_data = [
        ['id', 'calendar_id', 'weekday', 'start_time', 'end_time'],
        [1, 'testcal', 'Mon', 0.375, 0.708333],     # 09:00 to 17:00
        [2, 'testcal', 'Tue', 0.375, 0.708333],     # 09:00 to 17:00
        [3, 'testcal', 'Wed', 0.375, 0.708333],     # 09:00 to 17:00
        [4, 'testcal', 'Thu', 0.375, 0.708333],     # 09:00 to 17:00
        [5, 'testcal', 'Fri', 0.375, 0.708333],     # 09:00 to 17:00
    ]
    
    calendar_exceptions_data = [
        ['id', 'calendar_id', 'date', 'start_time', 'end_time', 'is_working']
    ]
    
    # Test Excel DATE function result (Excel serial number for 2025-01-01)
    # Excel serial number for 2025-01-01 is 45658
    start_datetime = 45658.375  # 2025-01-01 09:00 AM
    jobtime = 480  # 8 hours
    
    print(f"Testing with Excel serial datetime: {start_datetime}")
    print(f"Job time: {jobtime} minutes")
    
    try:
        result = calculate_working_completion_time(
            start_datetime=start_datetime,
            jobtime=jobtime,
            calendar_rules_data=calendar_rules_data,
            calendar_exceptions_data=calendar_exceptions_data,
            calendar_id="testcal"
        )
        
        print(f"Result: {result}")
        print(f"Result type: {type(result)}")
        
        # Convert result back to check
        if isinstance(result, (int, float)):
            # Convert Excel serial back to datetime to verify
            excel_epoch = datetime(1899, 12, 30)
            result_dt = excel_epoch + pd.Timedelta(days=result)
            print(f"Result as datetime: {result_dt}")
            
            # Check if this is the expected completion time
            # Starting at 2025-01-01 09:00, adding 8 hours of work should complete at 17:00 same day
            expected_completion = excel_epoch + pd.Timedelta(days=45658.708333)  # 2025-01-01 17:00
            print(f"Expected completion: {expected_completion}")
            
            if abs(result_dt - expected_completion) < pd.Timedelta(minutes=1):
                print("✓ Time calculation appears correct!")
            else:
                print("✗ Time calculation may be incorrect")
        else:
            print(f"✗ Unexpected result type or error: {result}")
            
    except Exception as e:
        print(f"✗ Error in calculation: {e}")

def test_mixed_time_formats():
    """Test calendar with mixed time formats (strings and Excel serial numbers)."""
    print("\n=== Testing Mixed Time Formats ===")
    
    # Mix of string times and Excel serial numbers
    calendar_rules_data = [
        ['id', 'calendar_id', 'weekday', 'start_time', 'end_time'],
        [1, 'mixedcal', 'Mon', '09:00', 0.708333],     # String start, Excel serial end
        [2, 'mixedcal', 'Tue', 0.375, '17:00'],        # Excel serial start, string end
        [3, 'mixedcal', 'Wed', '09:00', '17:00'],      # Both strings
        [4, 'mixedcal', 'Thu', 0.375, 0.708333],      # Both Excel serials
        [5, 'mixedcal', 'Fri', '09:00', '17:00'],      # Both strings
    ]
    
    calendar_exceptions_data = [
        ['id', 'calendar_id', 'date', 'start_time', 'end_time', 'is_working']
    ]
    
    start_datetime = 45658.375  # 2025-01-01 09:00 AM
    jobtime = 120  # 2 hours
    
    print(f"Testing mixed formats with datetime: {start_datetime}")
    print(f"Job time: {jobtime} minutes")
    
    try:
        result = calculate_working_completion_time(
            start_datetime=start_datetime,
            jobtime=jobtime,
            calendar_rules_data=calendar_rules_data,
            calendar_exceptions_data=calendar_exceptions_data,
            calendar_id="mixedcal"
        )
        
        print(f"Result: {result}")
        print(f"Result type: {type(result)}")
        
        if isinstance(result, (int, float)):
            # Convert Excel serial back to datetime to verify
            excel_epoch = datetime(1899, 12, 30)
            result_dt = excel_epoch + pd.Timedelta(days=result)
            print(f"Result as datetime: {result_dt}")
            print("✓ Mixed format handling appears to work!")
        else:
            print(f"✗ Unexpected result or error: {result}")
            
    except Exception as e:
        print(f"✗ Error in calculation: {e}")

if __name__ == "__main__":
    test_excel_time_conversion()
    test_calendar_with_excel_times()
    test_mixed_time_formats()