import pandas as pd
from datetime import datetime, timedelta, time, date
from typing import List, Dict, Tuple, Optional
import sys

class WorkingCalendarTester:
    """
    Independent implementation of working calendar logic for testing.
    This does NOT use any code from the target module.
    """
    
    def __init__(self, rules_file: str, exceptions_file: str):
        """Initialize with data files."""
        self.rules_df = pd.read_csv(rules_file)
        self.exceptions_df = pd.read_csv(exceptions_file)
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
                    if is_working:
                        # Working exception - this minute is working regardless of base
                        return True
                    else:
                        # Non-working exception - this minute is not working
                        return False
        
        return is_base_working
    
    def calculate_completion_time(self, start_dt: datetime, job_minutes: int, calendar_id: str) -> datetime:
        """
        Calculate completion time by checking each minute sequentially.
        This is the independent reference implementation.
        """
        if calendar_id not in self.rules:
            raise ValueError(f"Calendar {calendar_id} not found in rules")
        
        completed_minutes = 0
        current_dt = start_dt
        
        # Safety limit to prevent infinite loops
        max_iterations = job_minutes * 100  # Allow 100x the job duration
        iterations = 0
        
        while completed_minutes < job_minutes:
            iterations += 1
            if iterations > max_iterations:
                raise Exception(f"Exceeded maximum iterations ({max_iterations}). "
                              f"Possible infinite loop or insufficient working time available.")
            
            if self._is_working_minute(current_dt, calendar_id):
                completed_minutes += 1
                if completed_minutes >= job_minutes:
                    return current_dt + timedelta(minutes=1)
            
            current_dt += timedelta(minutes=1)
        
        return current_dt
    
    def get_available_calendars(self) -> List[str]:
        """Get list of available calendar IDs."""
        return list(self.rules.keys())
    
    def get_calendar_working_hours_per_week(self, calendar_id: str) -> float:
        """Calculate working hours per week for a calendar."""
        if calendar_id not in self.rules:
            return 0.0
        
        total_minutes = 0
        for weekday in range(7):
            periods = self.rules[calendar_id].get(weekday, [])
            for start_time, end_time in periods:
                start_dt = datetime.combine(date.today(), start_time)
                end_dt = datetime.combine(date.today(), end_time)
                total_minutes += (end_dt - start_dt).total_seconds() / 60
        
        return total_minutes / 60
    
    def has_exceptions_for_calendar(self, calendar_id: str) -> bool:
        """Check if calendar has any exceptions."""
        return calendar_id in self.exceptions and len(self.exceptions[calendar_id]) > 0
    
    def get_exception_count(self, calendar_id: str) -> Tuple[int, int]:
        """Get count of adding and deducting exceptions for a calendar."""
        if calendar_id not in self.exceptions:
            return 0, 0
        
        adding_count = 0
        deducting_count = 0
        
        for date_str, exceptions_list in self.exceptions[calendar_id].items():
            for _, _, is_working in exceptions_list:
                if is_working:
                    adding_count += 1
                else:
                    deducting_count += 1
        
        return adding_count, deducting_count
    
    def find_suitable_start_datetime(self, calendar_id: str, job_minutes: int) -> datetime:
        """Find a suitable start datetime that ensures the job can be completed."""
        # Start from a known working day
        start_date = datetime(2025, 1, 6)  # Monday
        
        # Try different start times to find a suitable one
        for hour in range(9, 12):  # Try morning hours
            for minute in [0, 30]:  # Try on the hour and half-hour
                candidate_start = start_date.replace(hour=hour, minute=minute)
                if self._is_working_minute(candidate_start, calendar_id):
                    return candidate_start
        
        raise Exception(f"Could not find suitable start datetime for calendar {calendar_id}")


def run_test_case(tester: WorkingCalendarTester, test_name: str, calendar_id: str,
                  job_minutes: int, start_dt: Optional[datetime] = None) -> bool:
    """Run a single test case."""
    print(f"\n=== {test_name} ===")
    print(f"Calendar: {calendar_id}")
    print(f"Job Duration: {job_minutes} minutes ({job_minutes/60:.1f} hours)")
    
    try:
        if start_dt is None:
            start_dt = tester.find_suitable_start_datetime(calendar_id, job_minutes)
        
        print(f"Start DateTime: {start_dt}")
        
        # Calculate completion time
        completion_dt = tester.calculate_completion_time(start_dt, job_minutes, calendar_id)
        print(f"Completion DateTime: {completion_dt}")
        
        # Calculate elapsed time
        elapsed_time = completion_dt - start_dt
        print(f"Elapsed Time: {elapsed_time}")
        
        print(f"✓ Test passed")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def main():
    """Run all test cases."""
    print("Working Calendar Test Suite")
    print("="*50)
    
    # Initialize tester
    tester = WorkingCalendarTester('data/calendar_rules.csv', 'data/calendar_exceptions.csv')
    
    # Get available calendars
    available_calendars = tester.get_available_calendars()
    print(f"Available calendars: {available_calendars}")
    
    # Print calendar info
    for cal in available_calendars:
        hours_per_week = tester.get_calendar_working_hours_per_week(cal)
        has_exceptions = tester.has_exceptions_for_calendar(cal)
        adding_count, deducting_count = tester.get_exception_count(cal)
        print(f"  {cal}: {hours_per_week} hours/week, "
              f"Exceptions: {has_exceptions} (Adding: {adding_count}, Deducting: {deducting_count})")
    
    test_results = []
    
    # Test 1: No exceptions, 120 mins job, each calendar
    print(f"\n{'='*50}")
    print("TEST 1: No exceptions, 120 minutes job, each calendar")
    print(f"{'='*50}")
    
    for calendar_id in available_calendars:
        has_exceptions = tester.has_exceptions_for_calendar(calendar_id)
        if not has_exceptions:
            result = run_test_case(tester, f"Test 1 - {calendar_id}", calendar_id, 120)
            test_results.append(result)
    
    # Test 2: Single exception adding time, 60 mins job, calC
    print(f"\n{'='*50}")
    print("TEST 2: Single exception adding time, 60 minutes job, calC")
    print(f"{'='*50}")
    
    adding_count, deducting_count = tester.get_exception_count('calC')
    if adding_count > 0:
        # Use a start time that will encounter an adding exception
        start_dt = datetime(2025, 1, 6, 11, 50)  # Should encounter 12:00-12:10 adding exception
        result = run_test_case(tester, "Test 2 - calC with adding exception", 'calC', 60, start_dt)
        test_results.append(result)
    else:
        print("✗ No adding exceptions found for calC")
        test_results.append(False)
    
    # Test 3: Single exception deducting time, calC
    print(f"\n{'='*50}")
    print("TEST 3: Single exception deducting time, calC")
    print(f"{'='*50}")
    
    if deducting_count > 0:
        # Use a start time that will encounter a deducting exception
        start_dt = datetime(2025, 1, 1, 9, 0)  # Should encounter 09:00-09:10 deducting exception
        result = run_test_case(tester, "Test 3 - calC with deducting exception", 'calC', 60, start_dt)
        test_results.append(result)
    else:
        print("✗ No deducting exceptions found for calC")
        test_results.append(False)
    
    # Test 4: 84 hours (5040 minutes) with multiple +/- exceptions, calC
    print(f"\n{'='*50}")
    print("TEST 4: 84 hours (5040 minutes) with multiple +/- exceptions, calC")
    print(f"{'='*50}")
    
    adding_count, deducting_count = tester.get_exception_count('calC')
    if adding_count >= 2 and deducting_count >= 2:
        # Start early to ensure we have enough time
        start_dt = datetime(2025, 1, 1, 9, 0)
        result = run_test_case(tester, "Test 4 - calC with multiple exceptions", 'calC', 5040, start_dt)
        test_results.append(result)
    else:
        print(f"✗ Insufficient exceptions for calC: Adding={adding_count}, Deducting={deducting_count}")
        test_results.append(False)
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed!")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)