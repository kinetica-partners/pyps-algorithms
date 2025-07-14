#!/usr/bin/env python3
"""
Comprehensive test suite for xlwings data type handling and calendar exceptions.
Core functionality tests without xlwings dependencies.
"""

import unittest
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd


def excel_boolean_to_python(excel_bool) -> bool:
    """Convert Excel boolean value to Python boolean."""
    if isinstance(excel_bool, bool):
        return excel_bool
    elif isinstance(excel_bool, (int, float)):
        return bool(excel_bool)  # 1 → True, 0 → False
    elif isinstance(excel_bool, str):
        # Handle string boolean values from Excel
        excel_bool_lower = excel_bool.lower().strip()
        if excel_bool_lower in ('true', '1', 'yes', 'on'):
            return True
        elif excel_bool_lower in ('false', '0', 'no', 'off', ''):
            return False
        else:
            # Default to False for unknown string values
            return False
    else:
        # Default to False for unknown types
        return False


def parse_time_string(time_string) -> Optional[time]:
    """Parse time string in HH:MM format to time object, handling Excel time serial numbers."""
    if isinstance(time_string, (int, float)):
        # Handle Excel time serial (fraction of day) with proper rounding
        total_seconds = round(time_string * 24 * 60 * 60)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return time(hours, minutes, seconds)
    
    try:
        if isinstance(time_string, str) and ':' in time_string:
            parts = time_string.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2]) if len(parts) > 2 else 0
            return time(hours, minutes, seconds)
    except (ValueError, IndexError):
        pass
    
    return None


def datetime_to_excel(dt):
    """Convert datetime to Excel serial number."""
    excel_epoch = datetime(1899, 12, 30)
    delta = dt - excel_epoch
    return float(delta.days) + (delta.seconds + delta.microseconds / 1e6) / 86400


def load_calendar_exceptions(exceptions_dataframe: pd.DataFrame) -> Dict[str, Dict[str, List[Tuple[time, time, bool]]]]:
    """Load calendar exceptions from dataframe into structured format."""
    if exceptions_dataframe.empty:
        return {}
    
    calendar_exceptions = {}
    
    for _, row in exceptions_dataframe.iterrows():
        calendar_id = str(row['calendar_id'])
        
        # Convert Excel date to proper date string format
        date_value = row['date']
        if isinstance(date_value, (int, float)):
            # Handle Excel serial number (days since 1899-12-30)
            if date_value > 20000:  # Modern dates
                excel_epoch = datetime(1899, 12, 30)
                date_obj = (excel_epoch + timedelta(days=date_value)).date()
                date_string = date_obj.strftime("%Y-%m-%d")
            else:
                # Fallback for unexpected small numbers
                date_string = str(date_value)
        else:
            # Handle string dates or other formats
            try:
                date_obj = pd.to_datetime(date_value).date()
                date_string = date_obj.strftime("%Y-%m-%d")
            except:
                date_string = str(date_value)
        
        calendar_exceptions.setdefault(calendar_id, {})
        
        # Add exception if both times are valid
        start_time_val = row['start_time']
        end_time_val = row['end_time']
        if pd.notnull(start_time_val) and pd.notnull(end_time_val):
            start_time = parse_time_string(start_time_val)
            end_time = parse_time_string(end_time_val)
            is_working_time = excel_boolean_to_python(row['is_working'])
            
            if start_time and end_time:
                calendar_exceptions[calendar_id].setdefault(date_string, []).append(
                    (start_time, end_time, is_working_time)
                )
    
    return calendar_exceptions


class TestExcelDataTypeHandling(unittest.TestCase):
    """Test Excel data type conversions that caused issues."""
    
    def test_excel_boolean_conversion_comprehensive(self):
        """Test all Excel boolean formats that caused problems."""
        # Test cases that caused the original bug
        test_cases = [
            # Python booleans
            (True, True),
            (False, False),
            
            # Excel numeric booleans
            (1, True),
            (0, False),
            (1.0, True),
            (0.0, False),
            
            # Excel string booleans (the main culprit)
            ("TRUE", True),
            ("FALSE", False),  # This was the bug - bool("FALSE") = True
            ("true", True),
            ("false", False),
            ("True", True),
            ("False", False),
            
            # Excel alternative formats
            ("1", True),
            ("0", False),
            ("yes", True),
            ("no", False),
            ("on", True),
            ("off", False),
            ("", False),
            
            # Unknown values should default to False for safety
            ("unknown", False),
            ("random", False),
            (None, False),
            ([], False),
            ({}, False),
        ]
        
        for input_val, expected in test_cases:
            with self.subTest(input_val=input_val):
                result = excel_boolean_to_python(input_val)
                self.assertEqual(result, expected, 
                    f"excel_boolean_to_python({repr(input_val)}) = {result}, expected {expected}")
    
    def test_excel_date_serial_conversion(self):
        """Test Excel date serial number conversion."""
        # Test cases based on Excel serial numbers from debug output
        test_cases = [
            # Excel serial 45658 = 2025-01-01 (from debug output)
            (45658, "2025-01-01"),
            (45659, "2025-01-02"),
            (45660, "2025-01-03"),
            
            # Known Excel dates
            (1, "1899-12-31"),  # Excel serial 1
            (36526, "2000-01-01"),  # Y2K (corrected)
            (43831, "2020-01-01"),  # Recent date
        ]
        
        for serial, expected_date_str in test_cases:
            with self.subTest(serial=serial):
                # Convert serial to date
                excel_epoch = datetime(1899, 12, 30)
                date_obj = (excel_epoch + timedelta(days=serial)).date()
                result = date_obj.strftime("%Y-%m-%d")
                self.assertEqual(result, expected_date_str,
                    f"Excel serial {serial} = {result}, expected {expected_date_str}")
    
    def test_excel_time_serial_conversion(self):
        """Test Excel time serial number conversion."""
        # Test cases based on debug output
        test_cases = [
            # From debug: 0.875 = 21:00:00
            (0.875, "21:00:00"),
            (0.881944444444444, "21:10:00"),  # From debug
            
            # Common Excel time values
            (0.0, "00:00:00"),
            (0.5, "12:00:00"),
            (0.25, "06:00:00"),
            (0.75, "18:00:00"),
            (0.375, "09:00:00"),  # 9 AM
        ]
        
        for serial, expected_time_str in test_cases:
            with self.subTest(serial=serial):
                result = parse_time_string(serial)
                expected_time = time.fromisoformat(expected_time_str)
                self.assertEqual(result, expected_time,
                    f"Excel time serial {serial} = {result}, expected {expected_time}")
    
    def test_datetime_excel_conversion(self):
        """Test datetime to Excel serial conversion."""
        test_cases = [
            (datetime(2025, 1, 1, 9, 0, 0), 45658.375),
            (datetime(2025, 1, 1, 21, 0, 0), 45658.875),
            (datetime(2025, 1, 2, 12, 10, 0), 45659.5069444444),
        ]
        
        for dt, expected_serial in test_cases:
            with self.subTest(dt=dt):
                excel_serial = datetime_to_excel(dt)
                self.assertAlmostEqual(excel_serial, expected_serial, places=5,
                    msg=f"datetime_to_excel({dt}) = {excel_serial}, expected {expected_serial}")


class TestCalendarExceptionsIntegration(unittest.TestCase):
    """Test calendar exceptions with real Excel data formats."""
    
    def setUp(self):
        """Set up test data matching Excel format from debug output."""
        # Calendar exceptions data (Excel format with serial numbers)
        self.calendar_exceptions_data = [
            ['id', 'calendar_id', 'date', 'start_time', 'end_time', 'is_working'],
            [1, 'calC', 45658, 0.875, 0.881944444444444, False],  # 2025-01-01 21:00-21:10 non-working
            [2, 'calC', 45659, 0.5, 0.50694444444444, False],     # 2025-01-02 12:00-12:10 non-working
            [3, 'calC', 45660, 0.5, 0.50694444444444, True],      # 2025-01-03 12:00-12:10 working
            [4, 'calC', 45661, 0.5, 0.51388888888888, True],      # 2025-01-04 12:00-12:20 working
        ]
    
    def test_load_calendar_exceptions_with_excel_data(self):
        """Test loading exceptions with Excel serial numbers and booleans."""
        # Create DataFrame from Excel data
        headers = self.calendar_exceptions_data[0]
        data = self.calendar_exceptions_data[1:]
        exceptions_df = pd.DataFrame(data, columns=headers)
        
        # Load exceptions
        exceptions = load_calendar_exceptions(exceptions_df)
        
        # Verify structure
        self.assertIn('calC', exceptions)
        self.assertEqual(len(exceptions['calC']), 4)  # 4 exception dates
        
        # Verify date conversion (Excel serial to YYYY-MM-DD)
        expected_dates = ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04']
        for expected_date in expected_dates:
            self.assertIn(expected_date, exceptions['calC'])
        
        # Verify specific exceptions
        # 2025-01-01: 21:00-21:10 non-working
        jan1_exceptions = exceptions['calC']['2025-01-01']
        self.assertEqual(len(jan1_exceptions), 1)
        start_time, end_time, is_working = jan1_exceptions[0]
        self.assertEqual(start_time, time(21, 0))
        self.assertEqual(end_time, time(21, 10))
        self.assertFalse(is_working)
        
        # 2025-01-03: 12:00-12:10 working
        jan3_exceptions = exceptions['calC']['2025-01-03']
        self.assertEqual(len(jan3_exceptions), 1)
        start_time, end_time, is_working = jan3_exceptions[0]
        self.assertEqual(start_time, time(12, 0))
        self.assertEqual(end_time, time(12, 10))
        self.assertTrue(is_working)


class TestPreviousTestFailures(unittest.TestCase):
    """Test cases that previous tests missed."""
    
    def test_boolean_string_edge_cases(self):
        """Test boolean string cases that weren't caught before."""
        # These are the cases that caused the original bug
        problematic_cases = [
            ("FALSE", False),  # bool("FALSE") incorrectly returns True
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
            ("off", False),
            ("", False),
        ]
        
        for input_val, expected in problematic_cases:
            with self.subTest(input_val=input_val):
                result = excel_boolean_to_python(input_val)
                self.assertEqual(result, expected,
                    f"CRITICAL: excel_boolean_to_python({repr(input_val)}) = {result}, expected {expected}")
    
    def test_date_format_mismatch(self):
        """Test the date format mismatch that caused exceptions not to work."""
        # This was the critical bug - Excel serial numbers stored as strings
        # but lookup expecting YYYY-MM-DD format
        
        # Create test data with Excel serial numbers
        test_data = [
            ['id', 'calendar_id', 'date', 'start_time', 'end_time', 'is_working'],
            [1, 'test', 45658, 0.5, 0.6, False],  # Excel serial number
        ]
        
        df = pd.DataFrame(test_data[1:], columns=test_data[0])
        exceptions = load_calendar_exceptions(df)
        
        # The key should be converted to YYYY-MM-DD format
        self.assertIn('test', exceptions)
        self.assertIn('2025-01-01', exceptions['test'])  # Not '45658'
        self.assertNotIn('45658', exceptions['test'])  # Should NOT be stored as string serial
    
    def test_time_precision_handling(self):
        """Test time precision issues that could cause problems."""
        # Test times with high precision from Excel
        test_cases = [
            (0.881944444444444, time(21, 10)),  # From debug output
            (0.50694444444444, time(12, 10)),
            (0.51388888888888, time(12, 20)),
        ]
        
        for excel_time, expected_time in test_cases:
            with self.subTest(excel_time=excel_time):
                result = parse_time_string(excel_time)
                self.assertEqual(result, expected_time,
                    f"Time precision issue: {excel_time} -> {result}, expected {expected_time}")


def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("Running comprehensive xlwings test suite...")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestExcelDataTypeHandling,
        TestCalendarExceptionsIntegration,
        TestPreviousTestFailures,
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASS' if success else 'FAIL'}")
    return success


if __name__ == '__main__':
    success = run_comprehensive_tests()
    exit(0 if success else 1)