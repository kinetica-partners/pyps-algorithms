#!/usr/bin/env python3
"""
Comprehensive test suite for xlwings data type handling and calendar exceptions.
Tests all the issues that were discovered during development.
"""

import unittest
from datetime import datetime, date, time, timedelta
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from working_calendar import (
    excel_boolean_to_python,
    parse_time_string,
    datetime_to_excel,
    load_calendar_exceptions,
    load_calendar_rules,
    build_working_intervals,
    calculate_working_completion_time
)
import pandas as pd


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
            (36526, "1999-12-31"),  # Y2K
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
        # Calendar rules data (Excel format)
        self.calendar_rules_data = [
            ['id', 'calendar_id', 'weekday', 'start_time', 'end_time'],
            [1, 'calC', 'Mon', '09:00', '17:00'],
            [2, 'calC', 'Tue', '09:00', '17:00'],
            [3, 'calC', 'Wed', '09:00', '17:00'],
            [4, 'calC', 'Thu', '09:00', '17:00'],
            [5, 'calC', 'Fri', '09:00', '17:00']
        ]
        
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
    
    def test_build_working_intervals_with_exceptions(self):
        """Test that working intervals properly apply exceptions."""
        # Load calendar data
        rules_df = pd.DataFrame(self.calendar_rules_data[1:], columns=self.calendar_rules_data[0])
        exceptions_df = pd.DataFrame(self.calendar_exceptions_data[1:], columns=self.calendar_exceptions_data[0])
        
        rules = load_calendar_rules(rules_df)
        exceptions = load_calendar_exceptions(exceptions_df)
        
        # Build working intervals
        start_date = date(2025, 1, 1)
        end_date = date(2025, 1, 4)
        intervals = build_working_intervals(rules, exceptions, 'calC', start_date, end_date)
        
        # Verify intervals exist
        self.assertGreater(len(intervals), 0)
        
        # Check that exceptions are applied
        # Find interval for 2025-01-01 21:00-21:10 (should be excluded due to non-working exception)
        jan1_evening_found = False
        for start_dt, end_dt, _ in intervals:
            if start_dt.date() == date(2025, 1, 1) and start_dt.time() == time(21, 0):
                jan1_evening_found = True
                break
        
        # The 21:00-21:10 period should be excluded due to non-working exception
        self.assertFalse(jan1_evening_found, "Non-working exception not applied")
    
    def test_full_integration_with_excel_data(self):
        """Test full integration with Excel data types."""
        # Test the main function with Excel data
        start_datetime = 45658.875  # 2025-01-01 21:00 as Excel serial
        jobtime = 60.0  # 1 hour
        
        result = calculate_working_completion_time(
            start_datetime,
            jobtime,
            self.calendar_rules_data,
            self.calendar_exceptions_data,
            'calC'
        )
        
        # Should return an Excel serial number
        self.assertIsInstance(result, float)
        self.assertGreater(result, 45658)  # Should be later than start date


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
    sys.exit(0 if success else 1)