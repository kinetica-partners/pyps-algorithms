import pandas as pd
from datetime import datetime, timedelta, time, date
from bisect import bisect_right
from typing import List, Dict, Tuple, Optional
import sys
import os
# Add src directory to path when running as main script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))

try:
    from .config import get_file_path, get_data_path
except ImportError:
    from config import get_file_path, get_data_path

# xlwings imports (for Excel integration)
import xlwings as xw
try:
    from xlwings import func, arg, script
except ImportError:
    # script may not be available in all xlwings versions
    from xlwings import func, arg
    script = None

# --- Constants ---
EXCEL_EPOCH = datetime(1899, 12, 30)
WEEKDAY_MAP = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
TIME_FORMAT = "%H:%M"
DATE_FORMAT = "%Y-%m-%d"
EXCEL_BOOL_TRUE = ('true', '1', 'yes', 'on')
EXCEL_BOOL_FALSE = ('false', '0', 'no', 'off', '')
MINUTES_PER_DAY = 1440
HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 60
SECONDS_PER_DAY = 86400

# --- Data Loading Helpers ---

def convert_excel_time(excel_time) -> str:
    """Convert Excel time serial number to HH:MM format."""
    if not isinstance(excel_time, (int, float)):
        return str(excel_time)
    
    total_minutes = round(excel_time * HOURS_PER_DAY * MINUTES_PER_HOUR)
    hours, minutes = divmod(total_minutes, MINUTES_PER_HOUR)
    return f"{hours:02d}:{minutes:02d}"

def convert_excel_boolean(excel_bool) -> bool:
    """Convert Excel boolean value to Python boolean."""
    if isinstance(excel_bool, bool):
        return excel_bool
    if isinstance(excel_bool, (int, float)):
        return bool(excel_bool)
    if isinstance(excel_bool, str):
        return excel_bool.lower().strip() in EXCEL_BOOL_TRUE
    return False

def parse_time_string(time_string) -> Optional[time]:
    """Parse time string in HH:MM format to time object, handling Excel time serial numbers."""
    if not pd.notnull(time_string) or time_string == '':
        return None
    
    try:
        time_str = convert_excel_time(time_string) if isinstance(time_string, (int, float)) else str(time_string)
        return datetime.strptime(time_str, TIME_FORMAT).time()
    except ValueError as e:
        raise ValueError(f"Invalid time format: {time_string}. Expected HH:MM format or Excel time serial number. Error: {e}")

def load_calendar_rules(rules_dataframe: pd.DataFrame) -> Dict[str, Dict[int, List[Tuple[time, time]]]]:
    """Load calendar rules from dataframe into structured format."""
    if rules_dataframe.empty:
        return {}
    
    calendar_rules = {}
    for _, row in rules_dataframe.iterrows():
        calendar_id = str(row['calendar_id']).lower()
        weekday_number = WEEKDAY_MAP[str(row['weekday'])]
        
        calendar_rules.setdefault(calendar_id, {day: [] for day in range(7)})
        
        # Add working period if both times are valid
        if pd.notnull(row['start_time']) and pd.notnull(row['end_time']):
            start_time = parse_time_string(row['start_time'])
            end_time = parse_time_string(row['end_time'])
            if start_time and end_time:
                calendar_rules[calendar_id][weekday_number].append((start_time, end_time))
    
    return calendar_rules

def load_calendar_exceptions(exceptions_dataframe: pd.DataFrame) -> Dict[str, Dict[str, List[Tuple[time, time, bool]]]]:
    """Load calendar exceptions from dataframe into structured format."""
    if exceptions_dataframe.empty:
        return {}
    
    calendar_exceptions = {}
    for _, row in exceptions_dataframe.iterrows():
        calendar_id = str(row['calendar_id']).lower()
        date_str = str(row['date'])
        is_working = convert_excel_boolean(row['is_working'])
        
        calendar_exceptions.setdefault(calendar_id, {})
        
        if is_working and pd.notnull(row['start_time']) and pd.notnull(row['end_time']):
            start_time = parse_time_string(row['start_time'])
            end_time = parse_time_string(row['end_time'])
            if start_time and end_time:
                calendar_exceptions[calendar_id].setdefault(date_str, []).append((start_time, end_time, True))
        elif not is_working:
            calendar_exceptions[calendar_id][date_str] = [(time(0, 0), time(0, 0), False)]
    
    return calendar_exceptions

# --- Core Precompute and Query Functions ---

def periods_overlap(p1_start: time, p1_end: time, p2_start: time, p2_end: time) -> bool:
    """Check if two time periods overlap."""
    return not (p1_end <= p2_start or p1_start >= p2_end)

def split_working_period_by_exception(
    working_period: Tuple[time, time],
    exception_period: Tuple[time, time]
) -> List[Tuple[time, time]]:
    """Split a working period by removing an exception period."""
    work_start, work_end = working_period
    exc_start, exc_end = exception_period
    
    if not periods_overlap(work_start, work_end, exc_start, exc_end):
        return [(work_start, work_end)]
    
    # Return non-overlapping parts
    result = []
    if work_start < exc_start:
        result.append((work_start, exc_start))
    if exc_end < work_end:
        result.append((exc_end, work_end))
    return result

def subtract_non_working_exceptions(
    working_periods: List[Tuple[time, time]],
    non_working_exceptions: List[Tuple[time, time]]
) -> List[Tuple[time, time]]:
    """Remove non-working exception periods from working periods."""
    result_periods = working_periods.copy()
    for exception_period in non_working_exceptions:
        result_periods = [
            split_period for period in result_periods
            for split_period in split_working_period_by_exception(period, exception_period)
        ]
    return result_periods

def add_non_overlapping_working_exceptions(
    working_periods: List[Tuple[time, time]],
    working_exceptions: List[Tuple[time, time]]
) -> List[Tuple[time, time]]:
    """Add working exceptions that don't overlap with existing working periods."""
    result_periods = working_periods.copy()
    for exc_start, exc_end in working_exceptions:
        if not any(periods_overlap(exc_start, exc_end, exist_start, exist_end)
                  for exist_start, exist_end in result_periods):
            result_periods.append((exc_start, exc_end))
    return result_periods

def get_working_periods_for_day(
    day: date,
    calendar_id: str,
    rules: Dict[str, Dict[int, List[Tuple[time, time]]]],
    exceptions: Dict[str, Dict[str, List[Tuple[time, time, bool]]]]
) -> List[Tuple[time, time]]:
    """Get working periods for a specific day, applying calendar rules and exceptions."""
    date_string = day.strftime("%Y-%m-%d")
    calendar_exceptions = exceptions.get(calendar_id, {})
    
    # Start with regular working periods for this day
    base_working_periods = rules.get(calendar_id, {}).get(day.weekday(), [])
    
    if date_string not in calendar_exceptions:
        # No exceptions, return regular periods
        return base_working_periods
    
    # There are exceptions for this day
    day_exceptions = calendar_exceptions[date_string]
    
    # Separate working and non-working exceptions
    working_exceptions = [(start, end) for start, end, is_working in day_exceptions if is_working]
    non_working_exceptions = [(start, end) for start, end, is_working in day_exceptions if not is_working]
    
    # Process exceptions
    result_periods = subtract_non_working_exceptions(base_working_periods, non_working_exceptions)
    result_periods = add_non_overlapping_working_exceptions(result_periods, working_exceptions)
    
    # Sort periods by start time
    result_periods.sort(key=lambda x: x[0])
    
    return result_periods

def build_working_intervals(
    rules: Dict[str, Dict[int, List[Tuple[time, time]]]],
    exceptions: Dict[str, Dict[str, List[Tuple[time, time, bool]]]],
    calendar_id: str,
    start_date: date,
    end_date: date
) -> List[Tuple[datetime, datetime, int]]:
    """Build working intervals for a date range with performance optimizations."""
    intervals = []
    total_minutes = 0
    
    # Pre-fetch calendar data and cache time calculations
    calendar_rules = rules.get(calendar_id, {})
    calendar_exceptions = exceptions.get(calendar_id, {})
    time_cache = {}
    
    def cached_time_diff(start_time: time, end_time: time) -> int:
        """Get cached time difference in minutes."""
        key = (start_time, end_time)
        if key not in time_cache:
            start_min = start_time.hour * MINUTES_PER_HOUR + start_time.minute
            end_min = end_time.hour * MINUTES_PER_HOUR + end_time.minute
            # Handle overnight shifts
            duration = (MINUTES_PER_DAY - start_min) + end_min if end_min <= start_min else end_min - start_min
            time_cache[key] = duration
        return time_cache[key]
    
    # Process all days
    current_date = start_date
    while current_date <= end_date:
        base_periods = calendar_rules.get(current_date.weekday(), [])
        date_string = current_date.strftime(DATE_FORMAT)
        
        # Apply exceptions if they exist
        if date_string in calendar_exceptions:
            day_exceptions = calendar_exceptions[date_string]
            working_exceptions = [(start, end) for start, end, is_working in day_exceptions if is_working]
            non_working_exceptions = [(start, end) for start, end, is_working in day_exceptions if not is_working]
            
            periods = subtract_non_working_exceptions(base_periods, non_working_exceptions)
            periods = add_non_overlapping_working_exceptions(periods, working_exceptions)
            periods.sort(key=lambda x: x[0])
        else:
            periods = base_periods
        
        # Convert to datetime intervals
        for start_time, end_time in periods:
            start_dt = datetime.combine(current_date, start_time)
            end_dt = datetime.combine(current_date, end_time)
            
            # Handle overnight shifts
            if end_time <= start_time:
                end_dt += timedelta(days=1)
            
            intervals.append((start_dt, end_dt, total_minutes))
            total_minutes += cached_time_diff(start_time, end_time)
        
        current_date += timedelta(days=1)
    
    return intervals

def add_working_minutes(
    start_dt: datetime,
    minutes_to_add: int,
    working_intervals: List[Tuple[datetime, datetime, int]]
) -> Optional[datetime]:
    """Add working minutes to a start datetime using precomputed working intervals."""
    if not working_intervals:
        return None
    
    # Find the interval containing or after start_dt
    interval_idx = next((i for i, (interval_start, interval_end, _) in enumerate(working_intervals)
                        if start_dt <= interval_end), -1)
    
    if interval_idx == -1:
        return None
    
    interval_start, interval_end, cumulative_at_start = working_intervals[interval_idx]
    
    # Calculate effective start and offset
    if start_dt < interval_start:
        offset = 0
        effective_start = interval_start
    else:
        offset = int((start_dt - interval_start).total_seconds() // 60)
        effective_start = start_dt
    
    # Check if we can finish within this interval
    minutes_in_interval = int((interval_end - effective_start).total_seconds() // 60)
    if minutes_to_add <= minutes_in_interval:
        return effective_start + timedelta(minutes=minutes_to_add)
    
    # Binary search for target interval
    target_cumulative = cumulative_at_start + offset + minutes_to_add
    left, right = 0, len(working_intervals) - 1
    target_idx = -1
    
    while left <= right:
        mid = (left + right) // 2
        mid_start, mid_end, mid_cumulative = working_intervals[mid]
        mid_duration = int((mid_end - mid_start).total_seconds() // 60)
        mid_cumulative_end = mid_cumulative + mid_duration
        
        if target_cumulative <= mid_cumulative_end:
            target_idx = mid
            right = mid - 1
        else:
            left = mid + 1
    
    if target_idx == -1:
        return None
    
    # Calculate final position
    target_start, _, target_cumulative_start = working_intervals[target_idx]
    minute_offset = target_cumulative - target_cumulative_start
    return target_start + timedelta(minutes=minute_offset)

def calculate_dynamic_buffer_days(
    minutes_to_add: int,
    calendar_rules: Dict[str, Dict[int, List[Tuple[time, time]]]],
    calendar_id: str
) -> int:
    """Calculate buffer days using heuristic based on actual calendar working hours."""
    calendar_data = calendar_rules.get(calendar_id, {})
    
    # Calculate weekly working minutes
    weekly_working_minutes = sum(
        (MINUTES_PER_DAY - start_time.hour * MINUTES_PER_HOUR - start_time.minute) +
        (end_time.hour * MINUTES_PER_HOUR + end_time.minute)
        if end_time.hour * MINUTES_PER_HOUR + end_time.minute <= start_time.hour * MINUTES_PER_HOUR + start_time.minute
        else (end_time.hour * MINUTES_PER_HOUR + end_time.minute) - (start_time.hour * MINUTES_PER_HOUR + start_time.minute)
        for weekday in range(7)
        for start_time, end_time in calendar_data.get(weekday, [])
    )
    
    # Handle edge case: no working time defined
    if weekly_working_minutes == 0:
        return max(365, minutes_to_add // MINUTES_PER_DAY)
    
    # Estimate calendar days needed
    estimated_days = minutes_to_add / (weekly_working_minutes / 7)
    
    # Apply safety margin based on job size
    if minutes_to_add < MINUTES_PER_DAY:
        safety_multiplier = 3.0  # Small jobs
    elif minutes_to_add < 43200:  # < 1 month of continuous work
        safety_multiplier = 2.0  # Medium jobs
    else:
        safety_multiplier = 1.5  # Large jobs
    
    safety_days = max(14, int(estimated_days * safety_multiplier))
    
    # Extra buffer for low-density calendars
    if weekly_working_minutes < 120:  # < 2 hours/week
        safety_days = max(safety_days, int(estimated_days * 5))
    
    # Extra buffer for sparse working days
    working_days_per_week = sum(1 for day in range(7) if calendar_data.get(day, []))
    if working_days_per_week <= 2:
        weeks_needed = int(estimated_days / 7) + 1
        safety_days = max(safety_days, weeks_needed * 14)
    
    # Cap maximum buffer
    max_buffer = min(3650, max(safety_days, minutes_to_add // 100))
    return min(safety_days, max_buffer)


# --- xlwings Lite Integration ---

def datetime_to_excel(dt: datetime) -> float:
    """Convert Python datetime to Excel date/time format (float)."""
    delta = dt - EXCEL_EPOCH
    return float(delta.days) + (delta.seconds + delta.microseconds / 1e6) / SECONDS_PER_DAY

def _convert_excel_datetime(excel_dt):
    """Convert Excel datetime input to Python datetime."""
    if isinstance(excel_dt, (int, float)):
        return datetime(1900, 1, 1) + timedelta(days=excel_dt - 2)
    return excel_dt

def _convert_jobtime_to_minutes(jobtime):
    """Convert jobtime to minutes (assumes hours if < 24, otherwise minutes)."""
    if isinstance(jobtime, (int, float)):
        return int(jobtime * 60) if jobtime < 24 else int(jobtime)
    return int(jobtime)

def _create_dataframe_from_data(data, default_func):
    """Create DataFrame from data or use default."""
    if data is None:
        default_data = default_func()
        return pd.DataFrame(default_data[1:], columns=default_data[0])
    return pd.DataFrame(data[1:], columns=data[0])

def get_default_calendar_rules():
    """Return default 9-5 Monday-Friday calendar rules."""
    return [
        ['id', 'calendar_id', 'weekday', 'start_time', 'end_time'],
        [1, 'default', 'Mon', '09:00', '17:00'],
        [2, 'default', 'Tue', '09:00', '17:00'],
        [3, 'default', 'Wed', '09:00', '17:00'],
        [4, 'default', 'Thu', '09:00', '17:00'],
        [5, 'default', 'Fri', '09:00', '17:00']
    ]

def get_default_calendar_exceptions():
    """Return default empty calendar exceptions."""
    return [['id', 'calendar_id', 'date', 'start_time', 'end_time', 'is_working']]

@func
def calculate_working_completion_time(
    start_datetime,
    jobtime,
    calendar_rules_data=None,
    calendar_exceptions_data=None,
    calendar_id="default"
):
    """Calculate working time completion datetime for Excel."""
    try:
        # Convert inputs
        start_dt = _convert_excel_datetime(start_datetime)
        minutes_to_add = _convert_jobtime_to_minutes(jobtime)
        
        # Load calendar data
        rules_df = _create_dataframe_from_data(calendar_rules_data, get_default_calendar_rules)
        exceptions_df = _create_dataframe_from_data(calendar_exceptions_data, get_default_calendar_exceptions)
        
        calendar_rules = load_calendar_rules(rules_df)
        calendar_exceptions = load_calendar_exceptions(exceptions_df)
        
        # Validate calendar_id or use first available
        calendar_id_lower = calendar_id.lower()
        if calendar_id_lower not in calendar_rules:
            if not calendar_rules:
                return "Error: No calendar rules found in data"
            calendar_id_lower = next(iter(calendar_rules.keys()))
        
        # Build working intervals and calculate completion time
        start_date = start_dt.date()
        buffer_days = calculate_dynamic_buffer_days(minutes_to_add, calendar_rules, calendar_id_lower)
        end_date = start_date + timedelta(days=buffer_days)
        
        working_intervals = build_working_intervals(
            calendar_rules, calendar_exceptions, calendar_id_lower, start_date, end_date
        )
        
        completion_dt = add_working_minutes(start_dt, minutes_to_add, working_intervals)
        
        if completion_dt is None:
            return f"Error: Cannot complete {minutes_to_add} minutes of work within the calculated time range"
        
        return datetime_to_excel(completion_dt)
        
    except Exception as e:
        return f"Error: {str(e)}"


def main(dataset='current'):
    """
    Main function to demonstrate working calendar functionality using configured paths.
    
    Args:
        dataset (str): Dataset name (current, simple, test, etc.) as defined in config.yaml
    """
    import pandas as pd
    from datetime import datetime
    
    try:
        # Load calendar data using portable configuration
        rules_file = get_file_path(dataset, 'calendar_rules')
        exceptions_file = get_file_path(dataset, 'calendar_exceptions')
        
        rules_df = pd.read_csv(rules_file)
        exceptions_df = pd.read_csv(exceptions_file)
        
        print("Working Calendar Demo")
        print("=" * 50)
        
        # Load calendar data
        rules = load_calendar_rules(rules_df)
        exceptions = load_calendar_exceptions(exceptions_df)
        
        print(f"Loaded {len(rules)} calendars with rules")
        print(f"Loaded {len(exceptions)} calendars with exceptions")
        
        # Demo calculation for each calendar
        start_datetime = datetime(2025, 1, 1, 14, 8)  # Monday 2:08 PM
        job_duration = 1000  # 1000 minutes

        print(f"\nDemo: Adding {job_duration} minutes to {start_datetime}")
        print("-" * 50)
        
        for calendar_id in rules.keys():
            print(f"\nCalendar '{calendar_id}':")
            
            # Build working intervals using dynamic buffer calculation
            start_date = start_datetime.date()
            buffer_days = calculate_dynamic_buffer_days(job_duration, rules, calendar_id)
            end_date = start_date + timedelta(days=buffer_days)
            
            intervals = build_working_intervals(
                rules, exceptions, calendar_id, start_date, end_date
            )
            
            if intervals:
                completion_time = add_working_minutes(start_datetime, job_duration, intervals)
                if completion_time:
                    duration = completion_time - start_datetime
                    print(f"  Start: {start_datetime}")
                    print(f"  End:   {completion_time}")
                    print(f"  Total elapsed: {duration}")
                else:
                    print(f"  Could not calculate completion time")
            else:
                print(f"  No working intervals found")
        
        print(f"\nWorking calendar demo completed successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: Calendar data files not found for dataset '{dataset}'")
        print(f"Expected files: calendar_rules.csv, calendar_exceptions.csv")
        print(f"Details: {e}")
    except Exception as e:
        print(f"Error during working calendar demo: {e}")


if __name__ == "__main__":
    main()