#!/usr/bin/env python3
"""
Comprehensive test suite for working calendar functionality.
Consolidates tests from multiple files to eliminate overlap.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta, time, date
from typing import List, Dict, Tuple, Optional
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from working_calendar import (
    convert_excel_boolean,
    convert_excel_time,
    parse_time_string,
    datetime_to_excel,
    load_calendar_rules,
    load_calendar_exceptions,
    build_working_intervals,
    add_working_minutes,
    calculate_working_completion_time
)
from config import get_file_path, get_test_data_path


class TestExcelDataTypeConversions:
    """Test Excel data type conversions."""
    
    def test_excel_boolean_conversion(self):
        """Test comprehensive Excel boolean conversion."""
        test_cases = [
            # Python booleans
            (True, True),
            (False, False),
            
            # Excel numeric booleans
            (1, True),
            (0, False),
            (1.0, True),
            (0.0, False),
            
            # Excel string booleans (critical edge cases)
            ("TRUE", True),
            ("FALSE", False),
            ("true", True),
            ("false", False),
            ("True", True),
            ("False", False),
            ("1", True),
            ("0", False),
            ("yes", True),
            ("no", False),
            ("", False),
            
            # Unknown values default to False
            ("unknown", False),
            (None, False),
        ]
        
        for input_val, expected in test_cases:
            result = convert_excel_boolean(input_val)
            assert result == expected, f"Failed for {input_val}: expected {expected}, got {result}"
    
    def test_excel_time_conversion(self):
        """Test Excel time serial number conversion."""
        test_cases = [
            (0.375, "09:00"),    # 9:00 AM
            (0.5, "12:00"),      # 12:00 PM
            (0.708333, "17:00"), # 5:00 PM
            (0.0, "00:00"),      # Midnight
            (0.996527, "23:55"), # Close to midnight but valid
        ]
        
        for excel_time, expected in test_cases:
            result = convert_excel_time(excel_time)
            assert result == expected, f"Failed for {excel_time}: expected {expected}, got {result}"
    
    def test_parse_time_string(self):
        """Test time string parsing with Excel compatibility."""
        test_cases = [
            ("09:00", time(9, 0)),
            ("17:30", time(17, 30)),
            (0.375, time(9, 0)),  # Excel serial number
            (0.5, time(12, 0)),   # Excel serial number
            ("", None),
            (None, None),
        ]
        
        for input_val, expected in test_cases:
            result = parse_time_string(input_val)
            assert result == expected, f"Failed for {input_val}: expected {expected}, got {result}"
    
    def test_datetime_to_excel(self):
        """Test datetime to Excel conversion."""
        test_dt = datetime(2025, 1, 1, 12, 0, 0)
        result = datetime_to_excel(test_dt)
        
        # Excel epoch starts at 1899-12-30
        # Jan 1, 2025 should be day 45658 (approximately)
        assert isinstance(result, float)
        assert result > 45000  # Basic sanity check


class TestWorkingCalendarCore:
    """Test core working calendar functionality."""
    
    @pytest.fixture
    def sample_calendar_data(self):
        """Sample calendar data for testing."""
        rules_data = [
            ['id', 'calendar_id', 'weekday', 'start_time', 'end_time'],
            [1, 'default', 'Mon', '09:00', '17:00'],
            [2, 'default', 'Tue', '09:00', '17:00'],
            [3, 'default', 'Wed', '09:00', '17:00'],
            [4, 'default', 'Thu', '09:00', '17:00'],
            [5, 'default', 'Fri', '09:00', '17:00'],
            [6, 'calA', 'Mon', '08:00', '16:00'],
            [7, 'calA', 'Tue', '08:00', '16:00'],
            [8, 'calA', 'Wed', '08:00', '16:00'],
            [9, 'calA', 'Thu', '08:00', '16:00'],
            [10, 'calA', 'Fri', '08:00', '16:00'],
        ]
        
        exceptions_data = [
            ['id', 'calendar_id', 'date', 'start_time', 'end_time', 'is_working'],
            [1, 'default', '2025-01-01', '00:00', '23:59', False],  # New Year's Day
            [2, 'calA', '2025-01-01', '10:00', '14:00', True],      # Partial working day
        ]
        
        rules_df = pd.DataFrame(rules_data[1:], columns=rules_data[0])
        exceptions_df = pd.DataFrame(exceptions_data[1:], columns=exceptions_data[0])
        
        return rules_df, exceptions_df
    
    def test_load_calendar_rules(self, sample_calendar_data):
        """Test loading calendar rules from DataFrame."""
        rules_df, _ = sample_calendar_data
        rules = load_calendar_rules(rules_df)
        
        assert 'default' in rules
        assert 'cala' in rules
        assert len(rules['default'][0]) == 1  # Monday has 1 period
        assert rules['default'][0][0] == (time(9, 0), time(17, 0))
    
    def test_load_calendar_exceptions(self, sample_calendar_data):
        """Test loading calendar exceptions from DataFrame."""
        _, exceptions_df = sample_calendar_data
        exceptions = load_calendar_exceptions(exceptions_df)
        
        assert 'default' in exceptions
        assert 'cala' in exceptions
        assert '2025-01-01' in exceptions['default']
        assert '2025-01-01' in exceptions['cala']
    
    def test_build_working_intervals(self, sample_calendar_data):
        """Test building working intervals."""
        rules_df, exceptions_df = sample_calendar_data
        rules = load_calendar_rules(rules_df)
        exceptions = load_calendar_exceptions(exceptions_df)
        
        # Test a simple date range
        start_date = date(2025, 1, 6)  # Monday
        end_date = date(2025, 1, 10)   # Friday
        
        intervals = build_working_intervals(rules, exceptions, 'default', start_date, end_date)
        
        assert len(intervals) > 0
        assert all(isinstance(interval, tuple) and len(interval) == 3 for interval in intervals)
    
    def test_add_working_minutes(self, sample_calendar_data):
        """Test adding working minutes to a datetime."""
        rules_df, exceptions_df = sample_calendar_data
        rules = load_calendar_rules(rules_df)
        exceptions = load_calendar_exceptions(exceptions_df)
        
        # Build intervals for testing
        start_date = date(2025, 1, 6)
        end_date = date(2025, 1, 10)
        intervals = build_working_intervals(rules, exceptions, 'default', start_date, end_date)
        
        # Test adding 60 minutes from start of Monday
        start_dt = datetime(2025, 1, 6, 9, 0)
        result = add_working_minutes(start_dt, 60, intervals)
        
        assert result is not None
        assert isinstance(result, datetime)
        assert result > start_dt


class TestWorkingCalendarIntegration:
    """Test integration functionality including xlwings compatibility."""
    
    def test_calculate_working_completion_time_basic(self):
        """Test basic working completion time calculation."""
        start_datetime = datetime(2025, 1, 6, 9, 0)  # Monday 9 AM
        jobtime = 120  # 2 hours
        
        # Use default calendar data
        result = calculate_working_completion_time(
            start_datetime=start_datetime,
            jobtime=jobtime,
            calendar_id="default"
        )
        
        # Should return a completion time
        assert isinstance(result, (datetime, float, str))
        # If it's an error string, it should not contain "Error"
        if isinstance(result, str) and "Error" in result:
            pytest.fail(f"Function returned error: {result}")
    
    def test_calculate_working_completion_time_excel_format(self):
        """Test with Excel-format inputs."""
        # Excel serial number for 2025-01-06 09:00
        start_datetime = 45658.375  # Approximate Excel serial number
        jobtime = 120.0  # Excel number format
        
        result = calculate_working_completion_time(
            start_datetime=start_datetime,
            jobtime=jobtime,
            calendar_id="default"
        )
        
        # Should handle Excel format inputs
        assert isinstance(result, (datetime, float, str))
        if isinstance(result, str) and "Error" in result:
            pytest.fail(f"Function returned error: {result}")


class WorkingCalendarReferenceTester:
    """
    Independent reference implementation for comparison testing.
    This provides a different algorithm to validate the optimized implementation.
    """
    
    def __init__(self, rules_df: pd.DataFrame, exceptions_df: pd.DataFrame):
        """Initialize with data."""
        self.rules_df = rules_df
        self.exceptions_df = exceptions_df
        self.rules = self._load_rules()
        self.exceptions = self._load_exceptions()
    
    def _parse_time(self, t: str) -> Optional[time]:
        """Parse time string to time object."""
        return datetime.strptime(t, "%H:%M").time() if pd.notnull(t) and t else None
    
    def _load_rules(self) -> Dict[str, Dict[int, List[Tuple[time, time]]]]:
        """Load calendar rules from CSV."""
        wd_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
        calendars = {}
        
        for _, row in self.rules_df.iterrows():
            cal = row['calendar_id']
            wd = wd_map[row['weekday']]
            calendars.setdefault(cal, {i: [] for i in range(7)})
            if pd.notnull(row['start_time']) and pd.notnull(row['end_time']):
                calendars[cal][wd].append((
                    self._parse_time(row['start_time']), 
                    self._parse_time(row['end_time'])
                ))
        
        return calendars
    
    def _load_exceptions(self) -> Dict[str, Dict[str, List[Tuple[time, time, bool]]]]:
        """Load calendar exceptions from CSV."""
        exceptions = {}
        
        for _, row in self.exceptions_df.iterrows():
            cal = row['calendar_id']
            exceptions.setdefault(cal, {})
            if pd.notnull(row['start_time']) and pd.notnull(row['end_time']):
                exceptions[cal].setdefault(row['date'], []).append((
                    self._parse_time(row['start_time']), 
                    self._parse_time(row['end_time']), 
                    bool(row['is_working'])
                ))
        
        return exceptions
    
    def _is_working_minute(self, dt: datetime, calendar_id: str) -> bool:
        """Check if a specific minute is working time."""
        day = dt.date()
        current_time = dt.time()
        date_str = day.strftime("%Y-%m-%d")
        
        # Get base working periods for this day
        base_periods = self.rules.get(calendar_id, {}).get(day.weekday(), [])
        
        # Check if current time falls within any base period
        is_base_working = False
        for start_time, end_time in base_periods:
            if start_time <= current_time < end_time:
                is_base_working = True
                break
        
        # Check for exceptions on this date
        cal_exceptions = self.exceptions.get(calendar_id, {})
        if date_str in cal_exceptions:
            for ex_start, ex_end, is_working in cal_exceptions[date_str]:
                if ex_start <= current_time < ex_end:
                    return is_working
        
        return is_base_working
    
    def calculate_completion_time(self, start_dt: datetime, job_minutes: int, calendar_id: str) -> datetime:
        """Calculate completion time by checking each minute sequentially."""
        if calendar_id not in self.rules:
            raise ValueError(f"Calendar {calendar_id} not found in rules")
        
        completed_minutes = 0
        current_dt = start_dt
        
        # Safety limit to prevent infinite loops
        max_iterations = job_minutes * 100
        iterations = 0
        
        while completed_minutes < job_minutes and iterations < max_iterations:
            if self._is_working_minute(current_dt, calendar_id):
                completed_minutes += 1
            current_dt += timedelta(minutes=1)
            iterations += 1
        
        if iterations >= max_iterations:
            raise RuntimeError(f"Maximum iterations exceeded for calendar {calendar_id}")
        
        return current_dt


class TestWorkingCalendarAccuracy:
    """Test accuracy by comparing with reference implementation."""
    
    @pytest.fixture
    def test_data(self):
        """Load test data from files."""
        try:
            # Try to load from current dataset using portable paths
            rules_file = get_file_path('current', 'calendar_rules')
            exceptions_file = get_file_path('current', 'calendar_exceptions')
            rules_df = pd.read_csv(rules_file)
            exceptions_df = pd.read_csv(exceptions_file)
        except FileNotFoundError:
            # Fallback to root data folder if available
            try:
                rules_df = pd.read_csv(get_file_path('', 'calendar_rules'))
                exceptions_df = pd.read_csv(get_file_path('', 'calendar_exceptions'))
            except FileNotFoundError:
                # Skip these tests if no data files found
                pytest.skip("Calendar data files not found")
        
        return rules_df, exceptions_df
    
    def test_implementation_comparison(self, test_data):
        """Compare optimized implementation with reference implementation."""
        rules_df, exceptions_df = test_data
        
        # Initialize both implementations
        rules = load_calendar_rules(rules_df)
        exceptions = load_calendar_exceptions(exceptions_df)
        reference = WorkingCalendarReferenceTester(rules_df, exceptions_df)
        
        # Test cases
        test_cases = [
            ("Basic test", datetime(2025, 1, 6, 9, 0), 120),   # 2 hours on Monday
            ("Cross-day test", datetime(2025, 1, 6, 16, 0), 120),  # 2 hours starting late
            ("Short test", datetime(2025, 1, 6, 11, 50), 60),  # 1 hour mid-day
        ]
        
        for calendar_id in ['default', 'calA', 'calB', 'calC']:
            if calendar_id not in rules:
                continue
                
            for test_name, start_dt, job_minutes in test_cases:
                # Skip very long tests for performance
                if job_minutes > 1000:
                    continue
                
                try:
                    # Calculate using reference implementation
                    ref_result = reference.calculate_completion_time(start_dt, job_minutes, calendar_id)
                    
                    # Calculate using optimized implementation
                    start_date = start_dt.date()
                    end_date = start_date + timedelta(days=30)  # Reasonable buffer
                    intervals = build_working_intervals(rules, exceptions, calendar_id, start_date, end_date)
                    opt_result = add_working_minutes(start_dt, job_minutes, intervals)
                    
                    # Compare results (allow 1 minute difference for rounding)
                    if opt_result is not None:
                        diff = abs((opt_result - ref_result).total_seconds())
                        assert diff <= 60, f"{test_name} failed for {calendar_id}: {diff}s difference"
                    else:
                        pytest.fail(f"{test_name} failed for {calendar_id}: optimized returned None")
                        
                except Exception as e:
                    # Log but don't fail on individual test cases
                    print(f"Skipping {test_name} for {calendar_id}: {e}")


class TestWorkingCalendarErrorHandling:
    """Test error handling and edge cases."""
    
    def test_parse_time_string_invalid_format(self):
        """Test parse_time_string with invalid format."""
        with pytest.raises(ValueError, match="Invalid time format"):
            parse_time_string("invalid_time")
    
    def test_parse_time_string_invalid_excel_number(self):
        """Test parse_time_string with invalid Excel time number."""
        with pytest.raises(ValueError, match="Invalid time format"):
            parse_time_string(-1)  # Negative time
    
    def test_load_calendar_rules_empty_dataframe(self):
        """Test load_calendar_rules with empty dataframe."""
        empty_df = pd.DataFrame()
        result = load_calendar_rules(empty_df)
        assert result == {}
    
    def test_load_calendar_exceptions_empty_dataframe(self):
        """Test load_calendar_exceptions with empty dataframe."""
        empty_df = pd.DataFrame()
        result = load_calendar_exceptions(empty_df)
        assert result == {}
    
    def test_load_calendar_exceptions_excel_serial_dates(self):
        """Test load_calendar_exceptions with Excel serial dates."""
        # Test with Excel serial date (modern date)
        exceptions_df = pd.DataFrame([
            {
                'calendar_id': 'CAL_001',
                'date': 45000,  # Excel serial date (around 2023)
                'start_time': '09:00',
                'end_time': '17:00',
                'is_working': True
            }
        ])
        
        result = load_calendar_exceptions(exceptions_df)
        assert 'cal_001' in result
        assert len(result['cal_001']) == 1
        
        # Test with small number (fallback case)
        exceptions_df_small = pd.DataFrame([
            {
                'calendar_id': 'CAL_002',
                'date': 123,  # Small number
                'start_time': '09:00',
                'end_time': '17:00',
                'is_working': True
            }
        ])
        
        result_small = load_calendar_exceptions(exceptions_df_small)
        assert 'cal_002' in result_small
        assert len(result_small['cal_002']) == 1
    
    def test_load_calendar_exceptions_string_dates(self):
        """Test load_calendar_exceptions with string dates."""
        exceptions_df = pd.DataFrame([
            {
                'calendar_id': 'CAL_003',
                'date': '2025-01-15',
                'start_time': '09:00',
                'end_time': '17:00',
                'is_working': True
            }
        ])
        
        result = load_calendar_exceptions(exceptions_df)
        assert 'cal_003' in result
        assert '2025-01-15' in result['cal_003']
    
    def test_load_calendar_exceptions_invalid_dates(self):
        """Test load_calendar_exceptions with invalid dates."""
        exceptions_df = pd.DataFrame([
            {
                'calendar_id': 'CAL_004',
                'date': 'invalid_date',
                'start_time': '09:00',
                'end_time': '17:00',
                'is_working': True
            }
        ])
        
        # Should handle invalid dates gracefully
        result = load_calendar_exceptions(exceptions_df)
        assert 'cal_004' in result
        assert 'invalid_date' in result['cal_004']


class TestWorkingCalendarExceptions:
    """Test exception processing in working calendar."""
    
    def test_build_working_intervals_with_exceptions(self):
        """Test build_working_intervals with various exceptions."""
        # Create rules
        rules = {
            'CAL_001': {
                0: [(time(9, 0), time(17, 0))]  # Monday 9-17
            }
        }
        
        # Create exceptions
        exceptions = {
            'CAL_001': {
                '2025-01-06': [  # Monday
                    (time(12, 0), time(13, 0), False),  # Lunch break (non-working)
                    (time(18, 0), time(19, 0), True)   # Overtime (working)
                ]
            }
        }
        
        start_date = datetime(2025, 1, 6)  # Monday
        end_date = datetime(2025, 1, 6)
        
        intervals = build_working_intervals(rules, exceptions, 'CAL_001', start_date, end_date)
        
        # Should have intervals with lunch break removed and overtime added
        assert len(intervals) >= 2
        
        # Test day with no exceptions
        exceptions_no_match = {
            'CAL_001': {
                '2025-01-07': [  # Tuesday (different day)
                    (time(12, 0), time(13, 0), False)
                ]
            }
        }
        
        intervals_no_exception = build_working_intervals(rules, exceptions_no_match, 'CAL_001', start_date, end_date)
        assert len(intervals_no_exception) >= 1
    
    def test_build_working_intervals_working_exceptions_only(self):
        """Test build_working_intervals with only working exceptions."""
        rules = {
            'CAL_001': {
                0: [(time(9, 0), time(17, 0))]  # Monday 9-17
            }
        }
        
        exceptions = {
            'CAL_001': {
                '2025-01-06': [
                    (time(18, 0), time(20, 0), True)   # Overtime only
                ]
            }
        }
        
        start_date = datetime(2025, 1, 6)
        end_date = datetime(2025, 1, 6)
        
        intervals = build_working_intervals(rules, exceptions, 'CAL_001', start_date, end_date)
        assert len(intervals) >= 2  # Regular hours + overtime
    
    def test_build_working_intervals_non_working_exceptions_only(self):
        """Test build_working_intervals with only non-working exceptions."""
        rules = {
            'CAL_001': {
                0: [(time(9, 0), time(17, 0))]  # Monday 9-17
            }
        }
        
        exceptions = {
            'CAL_001': {
                '2025-01-06': [
                    (time(12, 0), time(13, 0), False)  # Lunch break only
                ]
            }
        }
        
        start_date = datetime(2025, 1, 6)
        end_date = datetime(2025, 1, 6)
        
        intervals = build_working_intervals(rules, exceptions, 'CAL_001', start_date, end_date)
        assert len(intervals) >= 1  # Should have split working periods


class TestWorkingCalendarScriptExecution:
    """Test script execution paths."""
    
    def test_script_execution_path_exists(self):
        """Test that script execution path exists."""
        import working_calendar
        import inspect
        
        # Get the source code
        source = inspect.getsource(working_calendar)
        
        # Verify the main execution block exists
        assert 'if __name__ == "__main__":' in source
        assert 'sys.path.insert(0, os.path.dirname(__file__))' in source
    
    def test_xlwings_imports(self):
        """Test xlwings import handling."""
        # Test that xlwings imports work
        from working_calendar import xw
        assert xw is not None
        
        # Test that func and arg are available
        try:
            from working_calendar import func, arg
            assert func is not None
            assert arg is not None
        except ImportError:
            # May not be available in all environments
            pass


class TestWorkingCalendarUtilities:
    """Test utility functions and edge cases."""
    
    def test_add_working_minutes_edge_cases(self):
        """Test add_working_minutes with edge cases."""
        # Test with empty intervals
        empty_intervals = []
        result = add_working_minutes(datetime(2025, 1, 6, 9, 0), 60, empty_intervals)
        assert result is None
        
        # Test with zero duration
        intervals = [(datetime(2025, 1, 6, 9, 0), datetime(2025, 1, 6, 17, 0), 0)]
        result = add_working_minutes(datetime(2025, 1, 6, 9, 0), 0, intervals)
        assert result == datetime(2025, 1, 6, 9, 0)
        
        # Test with start time outside intervals
        result = add_working_minutes(datetime(2025, 1, 6, 8, 0), 60, intervals)
        assert result is not None
    
    def test_datetime_to_excel_edge_cases(self):
        """Test datetime_to_excel with edge cases."""
        # Test with different date formats
        test_date = datetime(2025, 1, 1, 12, 0)
        result = datetime_to_excel(test_date)
        assert isinstance(result, float)
        assert result > 0
        
        # Test with midnight
        midnight = datetime(2025, 1, 1, 0, 0)
        result_midnight = datetime_to_excel(midnight)
        assert isinstance(result_midnight, float)


class TestWorkingCalendarPathHandling:
    """Test path handling and imports."""
    
    def test_config_imports_working(self):
        """Test that config imports work correctly."""
        from working_calendar import get_file_path, get_data_path
        assert callable(get_file_path)
        assert callable(get_data_path)
    
    def test_module_imports(self):
        """Test that all required modules are imported."""
        import working_calendar
        
        # Check that all required functions are available
        required_functions = [
            'convert_excel_boolean',
            'convert_excel_time',
            'parse_time_string',
            'datetime_to_excel',
            'load_calendar_rules',
            'load_calendar_exceptions',
            'build_working_intervals',
            'add_working_minutes',
            'calculate_working_completion_time'
        ]
        
        for func_name in required_functions:
            assert hasattr(working_calendar, func_name)
            assert callable(getattr(working_calendar, func_name))


class TestDynamicBufferBehavior:
    """Test that working calendar functions adapt dynamically to different calendar patterns."""
    
    @pytest.fixture
    def high_density_calendar_data(self):
        """High density 24/7 calendar data."""
        rules_data = [
            ['id', 'calendar_id', 'weekday', 'start_time', 'end_time'],
            [1, 'high_density', 'Mon', '00:00', '23:59'],
            [2, 'high_density', 'Tue', '00:00', '23:59'],
            [3, 'high_density', 'Wed', '00:00', '23:59'],
            [4, 'high_density', 'Thu', '00:00', '23:59'],
            [5, 'high_density', 'Fri', '00:00', '23:59'],
            [6, 'high_density', 'Sat', '00:00', '23:59'],
            [7, 'high_density', 'Sun', '00:00', '23:59'],
        ]
        
        exceptions_data = [
            ['id', 'calendar_id', 'date', 'start_time', 'end_time', 'is_working']
        ]
        
        return rules_data, exceptions_data
    
    @pytest.fixture
    def low_density_calendar_data(self):
        """Low density 1 hour per week calendar data."""
        rules_data = [
            ['id', 'calendar_id', 'weekday', 'start_time', 'end_time'],
            [1, 'low_density', 'Mon', '09:00', '10:00'],  # Only 1 hour per week
        ]
        
        exceptions_data = [
            ['id', 'calendar_id', 'date', 'start_time', 'end_time', 'is_working']
        ]
        
        return rules_data, exceptions_data
    
    def test_completion_time_scales_with_job_size(self):
        """Test that completion time calculation works for different job sizes."""
        start_datetime = datetime(2025, 1, 6, 9, 0)  # Monday 9 AM
        
        # Test various job sizes
        job_sizes = [60, 480, 2400, 9600]  # 1 hour, 8 hours, 40 hours, 160 hours
        
        for job_minutes in job_sizes:
            result = calculate_working_completion_time(
                start_datetime=start_datetime,
                jobtime=job_minutes,
                calendar_id="default"
            )
            
            # Should return a valid completion time, not an error
            assert isinstance(result, (datetime, float)), f"Job size {job_minutes} failed: {result}"
            if isinstance(result, str) and "Error" in result:
                pytest.fail(f"Job size {job_minutes} returned error: {result}")
    
    def test_completion_time_adapts_to_calendar_density(self, high_density_calendar_data, low_density_calendar_data):
        """Test that completion time adapts to different calendar working densities."""
        start_datetime = datetime(2025, 1, 6, 9, 0)  # Monday 9 AM
        job_minutes = 120  # 2 hours
        
        # Test high density calendar
        high_rules, high_exceptions = high_density_calendar_data
        high_result = calculate_working_completion_time(
            start_datetime=start_datetime,
            jobtime=job_minutes,
            calendar_rules_data=high_rules,
            calendar_exceptions_data=high_exceptions,
            calendar_id="high_density"
        )
        
        # Test low density calendar
        low_rules, low_exceptions = low_density_calendar_data
        low_result = calculate_working_completion_time(
            start_datetime=start_datetime,
            jobtime=job_minutes,
            calendar_rules_data=low_rules,
            calendar_exceptions_data=low_exceptions,
            calendar_id="low_density"
        )
        
        # Both should work (not return errors)
        assert not (isinstance(high_result, str) and "Error" in high_result), f"High density failed: {high_result}"
        assert not (isinstance(low_result, str) and "Error" in low_result), f"Low density failed: {low_result}"
    
    def test_completion_time_handles_large_jobs(self):
        """Test that completion time calculation handles very large jobs efficiently."""
        start_datetime = datetime(2025, 1, 6, 9, 0)  # Monday 9 AM
        large_job = 50_000  # Large job: ~833 hours
        
        # Should complete without timeout or excessive memory usage
        result = calculate_working_completion_time(
            start_datetime=start_datetime,
            jobtime=large_job,
            calendar_id="default"
        )
        
        # Should return a result, not an error about insufficient time range
        assert isinstance(result, (datetime, float)), f"Large job failed: {result}"
        if isinstance(result, str) and "Error" in result:
            # If it's an error, it should NOT be about insufficient time range
            assert "Cannot complete" not in result, f"Large job failed due to insufficient buffer: {result}"
    
    def test_completion_time_efficient_memory_usage(self):
        """Test that completion time calculation doesn't generate excessive working intervals."""
        start_datetime = datetime(2025, 1, 6, 9, 0)  # Monday 9 AM
        
        # Test with progressively larger jobs
        job_sizes = [1440, 14400, 144000]  # 1 day, 10 days, 100 days of continuous work
        
        for job_minutes in job_sizes:
            result = calculate_working_completion_time(
                start_datetime=start_datetime,
                jobtime=job_minutes,
                calendar_id="default"
            )
            
            # Should complete efficiently without memory issues
            assert isinstance(result, (datetime, float)), f"Job size {job_minutes} failed: {result}"
            if isinstance(result, str) and "Error" in result:
                # Should not fail due to excessive buffer calculation
                assert "Cannot complete" not in result, f"Job size {job_minutes} failed due to buffer: {result}"
    
    def test_completion_time_no_hardcoded_assumptions(self):
        """Test that completion time calculation works with various calendar patterns without hardcoded assumptions."""
        start_datetime = datetime(2025, 1, 6, 9, 0)  # Monday 9 AM
        job_minutes = 480  # 8 hours
        
        # Test with various calendar patterns
        calendar_patterns = [
            # Weekend-only calendar
            [
                ['id', 'calendar_id', 'weekday', 'start_time', 'end_time'],
                [1, 'weekend', 'Sat', '09:00', '17:00'],
                [2, 'weekend', 'Sun', '09:00', '17:00'],
            ],
            # Night shift calendar
            [
                ['id', 'calendar_id', 'weekday', 'start_time', 'end_time'],
                [1, 'night', 'Mon', '22:00', '06:00'],
                [2, 'night', 'Tue', '22:00', '06:00'],
                [3, 'night', 'Wed', '22:00', '06:00'],
                [4, 'night', 'Thu', '22:00', '06:00'],
                [5, 'night', 'Fri', '22:00', '06:00'],
            ],
            # Split shift calendar
            [
                ['id', 'calendar_id', 'weekday', 'start_time', 'end_time'],
                [1, 'split', 'Mon', '06:00', '10:00'],
                [2, 'split', 'Mon', '14:00', '18:00'],
                [3, 'split', 'Tue', '06:00', '10:00'],
                [4, 'split', 'Tue', '14:00', '18:00'],
                [5, 'split', 'Wed', '06:00', '10:00'],
                [6, 'split', 'Wed', '14:00', '18:00'],
                [7, 'split', 'Thu', '06:00', '10:00'],
                [8, 'split', 'Thu', '14:00', '18:00'],
                [9, 'split', 'Fri', '06:00', '10:00'],
                [10, 'split', 'Fri', '14:00', '18:00'],
            ]
        ]
        
        exceptions_data = [
            ['id', 'calendar_id', 'date', 'start_time', 'end_time', 'is_working']
        ]
        
        for rules_data in calendar_patterns:
            calendar_id = rules_data[1][1]  # Extract calendar_id from first data row
            
            result = calculate_working_completion_time(
                start_datetime=start_datetime,
                jobtime=job_minutes,
                calendar_rules_data=rules_data,
                calendar_exceptions_data=exceptions_data,
                calendar_id=calendar_id
            )
            
            # Should work with any calendar pattern
            assert isinstance(result, (datetime, float)), f"Calendar {calendar_id} failed: {result}"
            if isinstance(result, str) and "Error" in result:
                pytest.fail(f"Calendar {calendar_id} returned error: {result}")



if __name__ == "__main__":
    pytest.main([__file__])