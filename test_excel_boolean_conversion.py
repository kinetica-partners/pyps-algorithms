#!/usr/bin/env python3
"""Test Excel boolean conversion in working calendar."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from working_calendar import excel_boolean_to_python, calculate_working_completion_time
from datetime import datetime
import pandas as pd

def test_excel_boolean_conversion():
    """Test Excel boolean conversion function."""
    print("=== Testing Excel Boolean Conversion ===")
    
    # Test different boolean formats that Excel might send
    test_cases = [
        (True, True),           # Python boolean True
        (False, False),         # Python boolean False
        (1, True),              # Excel numeric True
        (0, False),             # Excel numeric False
        (1.0, True),            # Excel float True
        (0.0, False),           # Excel float False
        ("TRUE", True),         # Excel string TRUE
        ("FALSE", False),       # Excel string FALSE
        ("true", True),         # Lowercase true
        ("false", False),       # Lowercase false
        ("True", True),         # Title case True
        ("False", False),       # Title case False
        ("1", True),            # String "1"
        ("0", False),           # String "0"
        ("YES", True),          # Alternative yes
        ("NO", False),          # Alternative no
        ("", False),            # Empty string
        ("unknown", False),     # Unknown string (should default to False)
    ]
    
    print("\nTesting excel_boolean_to_python():")
    for input_val, expected in test_cases:
        result = excel_boolean_to_python(input_val)
        status = "✓" if result == expected else "✗"
        print(f"{status} {repr(input_val)} -> {result} (expected: {expected})")

def test_calendar_with_excel_boolean_exceptions():
    """Test calendar function with Excel boolean exceptions."""
    print("\n=== Testing Calendar with Excel Boolean Exceptions ===")
    
    # Create test calendar data with working hours
    calendar_rules_data = [
        ['id', 'calendar_id', 'weekday', 'start_time', 'end_time'],
        [1, 'testcal', 'Mon', '09:00', '17:00'],
        [2, 'testcal', 'Tue', '09:00', '17:00'],
        [3, 'testcal', 'Wed', '09:00', '17:00'],
        [4, 'testcal', 'Thu', '09:00', '17:00'],
        [5, 'testcal', 'Fri', '09:00', '17:00'],
    ]
    
    # Test calendar exceptions with different boolean formats
    test_cases = [
        ("Excel numeric boolean", [
            ['id', 'calendar_id', 'date', 'start_time', 'end_time', 'is_working'],
            [1, 'testcal', '2025-01-01', '12:00', '13:00', 0],  # Lunch break (non-working)
            [2, 'testcal', '2025-01-01', '10:00', '11:00', 1],  # Special working period
        ]),
        ("Excel string boolean", [
            ['id', 'calendar_id', 'date', 'start_time', 'end_time', 'is_working'],
            [1, 'testcal', '2025-01-01', '12:00', '13:00', 'FALSE'],  # Lunch break
            [2, 'testcal', '2025-01-01', '10:00', '11:00', 'TRUE'],   # Special working period
        ]),
        ("Mixed boolean formats", [
            ['id', 'calendar_id', 'date', 'start_time', 'end_time', 'is_working'],
            [1, 'testcal', '2025-01-01', '12:00', '13:00', 0],       # Numeric False
            [2, 'testcal', '2025-01-01', '10:00', '11:00', 'TRUE'],  # String True
            [3, 'testcal', '2025-01-01', '14:00', '15:00', False],   # Python False
        ]),
    ]
    
    for test_name, calendar_exceptions_data in test_cases:
        print(f"\nTesting {test_name}:")
        
        # Test with a job that would normally take 8 hours (480 minutes)
        # Starting at 09:00 on 2025-01-01
        start_datetime = 45658.375  # 2025-01-01 09:00 AM
        jobtime = 480  # 8 hours
        
        try:
            result = calculate_working_completion_time(
                start_datetime=start_datetime,
                jobtime=jobtime,
                calendar_rules_data=calendar_rules_data,
                calendar_exceptions_data=calendar_exceptions_data,
                calendar_id="testcal"
            )
            
            if isinstance(result, (int, float)):
                # Convert Excel serial back to datetime to check
                excel_epoch = datetime(1899, 12, 30)
                result_dt = excel_epoch + pd.Timedelta(days=result)
                print(f"  ✓ Result: {result} -> {result_dt}")
            else:
                print(f"  ✗ Unexpected result: {result}")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")

if __name__ == "__main__":
    test_excel_boolean_conversion()
    test_calendar_with_excel_boolean_exceptions()