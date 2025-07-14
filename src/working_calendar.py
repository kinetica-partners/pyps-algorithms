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

# --- Data Loading Helpers ---

def excel_time_to_string(excel_time) -> str:
    """Convert Excel time serial number to HH:MM format."""
    if isinstance(excel_time, (int, float)):
        # Excel time is fraction of a day (0.375 = 9:00 AM)
        # Add small epsilon to handle floating point precision issues
        total_minutes = round(excel_time * 24 * 60)
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours:02d}:{minutes:02d}"
    else:
        # Already a string, return as-is
        return str(excel_time)

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
    if not pd.notnull(time_string) or time_string == '':
        return None
    
    try:
        # If it's a number (Excel time serial), convert to HH:MM first
        if isinstance(time_string, (int, float)):
            converted_time = excel_time_to_string(time_string)
            return datetime.strptime(converted_time, "%H:%M").time()
        
        # Parse as HH:MM format
        time_string = str(time_string)
        return datetime.strptime(time_string, "%H:%M").time()
    except ValueError as e:
        raise ValueError(f"Invalid time format: {time_string}. Expected HH:MM format or Excel time serial number. Error: {e}")

def load_calendar_rules(rules_dataframe: pd.DataFrame) -> Dict[str, Dict[int, List[Tuple[time, time]]]]:
    """Load calendar rules from dataframe into structured format."""
    if rules_dataframe.empty:
        return {}
    
    weekday_to_number = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
    calendar_rules = {}
    
    for _, row in rules_dataframe.iterrows():
        calendar_id = str(row['calendar_id'])
        weekday_number = weekday_to_number[str(row['weekday'])]
        
        # Initialize calendar if not exists
        calendar_rules.setdefault(calendar_id, {day: [] for day in range(7)})
        
        # Add working period if both times are valid
        start_time_val = row['start_time']
        end_time_val = row['end_time']
        if pd.notnull(start_time_val) and pd.notnull(end_time_val):
            start_time = parse_time_string(start_time_val)
            end_time = parse_time_string(end_time_val)
            if start_time and end_time:
                calendar_rules[calendar_id][weekday_number].append((start_time, end_time))
    
    return calendar_rules

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

# --- Core Precompute and Query Functions ---

def periods_overlap(period1_start: time, period1_end: time, period2_start: time, period2_end: time) -> bool:
    """Check if two time periods overlap."""
    return not (period1_end <= period2_start or period1_start >= period2_end)

def split_working_period_by_exception(
    working_period: Tuple[time, time],
    exception_period: Tuple[time, time]
) -> List[Tuple[time, time]]:
    """Split a working period by removing an exception period."""
    work_start, work_end = working_period
    exception_start, exception_end = exception_period
    
    if not periods_overlap(work_start, work_end, exception_start, exception_end):
        # No overlap, return original period
        return [(work_start, work_end)]
    
    # There's overlap, split the working period
    result = []
    
    # Add the part before the exception period
    if work_start < exception_start:
        result.append((work_start, exception_start))
    
    # Add the part after the exception period
    if exception_end < work_end:
        result.append((exception_end, work_end))
    
    return result

def subtract_non_working_exceptions(
    working_periods: List[Tuple[time, time]],
    non_working_exceptions: List[Tuple[time, time]]
) -> List[Tuple[time, time]]:
    """Remove non-working exception periods from working periods."""
    result_periods = working_periods.copy()
    
    for exception_period in non_working_exceptions:
        new_periods = []
        for working_period in result_periods:
            new_periods.extend(split_working_period_by_exception(working_period, exception_period))
        result_periods = new_periods
    
    return result_periods

def add_non_overlapping_working_exceptions(
    working_periods: List[Tuple[time, time]],
    working_exceptions: List[Tuple[time, time]]
) -> List[Tuple[time, time]]:
    """Add working exceptions that don't overlap with existing working periods."""
    result_periods = working_periods.copy()
    
    for exception_start, exception_end in working_exceptions:
        # Check if this working exception overlaps with any existing working period
        has_overlap = any(
            periods_overlap(exception_start, exception_end, existing_start, existing_end)
            for existing_start, existing_end in result_periods
        )
        
        if not has_overlap:
            # No overlap, add this working exception
            result_periods.append((exception_start, exception_end))
    
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
    
    # Pre-fetch calendar data to avoid repeated lookups
    calendar_rules = rules.get(calendar_id, {})
    calendar_exceptions = exceptions.get(calendar_id, {})
    
    # Pre-calculate common time differences to avoid repeated calculations
    time_to_minutes_cache = {}
    
    def get_time_difference_minutes(start_time: time, end_time: time) -> int:
        """Get cached time difference in minutes."""
        key = (start_time, end_time)
        if key not in time_to_minutes_cache:
            # Calculate using direct time arithmetic instead of datetime
            start_minutes = start_time.hour * 60 + start_time.minute
            end_minutes = end_time.hour * 60 + end_time.minute
            time_to_minutes_cache[key] = end_minutes - start_minutes
        return time_to_minutes_cache[key]
    
    # Batch process all days
    current_date = start_date
    while current_date <= end_date:
        # Get base working periods for this weekday
        base_periods = calendar_rules.get(current_date.weekday(), [])
        
        # Check for exceptions on this specific date (cache date string)
        date_string = current_date.strftime("%Y-%m-%d")
        if date_string in calendar_exceptions:
            # Apply exceptions to get final periods
            day_exceptions = calendar_exceptions[date_string]
            working_exceptions = [(start, end) for start, end, is_working in day_exceptions if is_working]
            non_working_exceptions = [(start, end) for start, end, is_working in day_exceptions if not is_working]
            
            # Process exceptions
            periods = subtract_non_working_exceptions(base_periods, non_working_exceptions)
            periods = add_non_overlapping_working_exceptions(periods, working_exceptions)
            periods.sort(key=lambda x: x[0])
        else:
            # No exceptions, use base periods
            periods = base_periods
        
        # Convert to datetime intervals and add to result
        for start_time, end_time in periods:
            start_dt = datetime.combine(current_date, start_time)
            end_dt = datetime.combine(current_date, end_time)
            intervals.append((start_dt, end_dt, total_minutes))
            # Use cached time calculation instead of datetime arithmetic
            total_minutes += get_time_difference_minutes(start_time, end_time)
        
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
    idx = -1
    for i, (iv_start, iv_end, cumu) in enumerate(working_intervals):
        if start_dt <= iv_end:  # start_dt is within or before this interval
            idx = i
            break
    
    if idx == -1:
        # start_dt is after all intervals
        return None
    
    interval_start, interval_end, cumu_at_start = working_intervals[idx]
    
    # If start_dt is before the interval, move to the start of the interval
    if start_dt < interval_start:
        offset = 0
        effective_start = interval_start
    else:
        offset = int((start_dt - interval_start).total_seconds() // 60)
        effective_start = start_dt
    
    # Check if we can finish within this interval
    minutes_in_this_interval = int((interval_end - effective_start).total_seconds() // 60)
    if minutes_to_add <= minutes_in_this_interval:
        return effective_start + timedelta(minutes=minutes_to_add)
    
    # We need to go to subsequent intervals
    # Calculate the absolute target cumulative minutes from the very beginning
    current_position = cumu_at_start + offset
    target_cumu = current_position + minutes_to_add
    
    # Find the interval that contains the target cumulative minutes using direct binary search
    left, right = 0, len(working_intervals) - 1
    target_idx = -1
    
    while left <= right:
        mid = (left + right) // 2
        iv_start, iv_end, cumu_at_iv_start = working_intervals[mid]
        iv_duration = int((iv_end - iv_start).total_seconds() // 60)
        cumu_at_iv_end = cumu_at_iv_start + iv_duration
        
        if target_cumu <= cumu_at_iv_end:
            target_idx = mid
            right = mid - 1
        else:
            left = mid + 1
    
    if target_idx == -1:
        # Target is beyond all intervals
        return None
    
    # Found the target interval
    iv_start, iv_end, cumu_at_iv_start = working_intervals[target_idx]
    minute_offset = target_cumu - cumu_at_iv_start
    return iv_start + timedelta(minutes=minute_offset)

# --- xlwings Lite Integration ---

def datetime_to_excel(dt):
    """Convert Python datetime to Excel date/time format (float)."""
    # Excel's epoch starts at 1899-12-30
    excel_epoch = datetime(1899, 12, 30)
    delta = dt - excel_epoch
    return float(delta.days) + (delta.seconds + delta.microseconds / 1e6) / 86400

@func
def calculate_working_completion_time(
        start_datetime,
        jobtime,
        calendar_rules_data=None,
        calendar_exceptions_data=None,
        calendar_id="default"
    ):
        """
        Calculate working time completion datetime for Excel (xlwings lite compatible).
        
        Args:
            start_datetime: Excel datetime format start time
            jobtime: Job duration in minutes
            calendar_rules_data: Excel range with calendar rules data (2D array)
            calendar_exceptions_data: Excel range with calendar exceptions data (2D array)
            calendar_id: Calendar ID to use (default: "default")
            
        Returns:
            Excel datetime format completion time
        """
        try:
            # Convert Excel datetime to Python datetime with proper handling
            if isinstance(start_datetime, str):
                # Handle string datetime
                try:
                    start_dt = pd.to_datetime(start_datetime)
                    if hasattr(start_dt, 'to_pydatetime'):
                        start_dt = start_dt.to_pydatetime()
                except Exception as e:
                    return f"Error: Cannot convert start_datetime string '{start_datetime}' to datetime: {str(e)}"
            elif isinstance(start_datetime, (int, float)):
                # Handle Excel serial number (common when Excel sends DATE() function results)
                try:
                    # Check if this looks like an Excel serial number (days since 1899-12-30)
                    # Excel serial numbers for modern dates are typically 20000+ (1954+)
                    if start_datetime > 20000:
                        # Treat as Excel serial number
                        excel_epoch = datetime(1899, 12, 30)
                        start_dt = excel_epoch + timedelta(days=start_datetime)
                    else:
                        # Treat as Unix timestamp
                        start_dt = pd.to_datetime(start_datetime, unit='s')
                        if hasattr(start_dt, 'to_pydatetime'):
                            start_dt = start_dt.to_pydatetime()
                except Exception as e:
                    return f"Error: Cannot convert start_datetime number '{start_datetime}' to datetime: {str(e)}"
            elif isinstance(start_datetime, datetime):
                # Already a datetime
                start_dt = start_datetime
            elif hasattr(start_datetime, 'date') and hasattr(start_datetime, 'time'):
                # Already a datetime-like object
                start_dt = start_datetime
            elif hasattr(start_datetime, 'date'):
                # Date object - convert to datetime at midnight
                start_dt = datetime.combine(start_datetime.date(), time())
            else:
                # Try to convert whatever Excel sent us
                try:
                    start_dt = pd.to_datetime(start_datetime)
                    if hasattr(start_dt, 'to_pydatetime'):
                        start_dt = start_dt.to_pydatetime()
                except Exception as e:
                    return f"Error: Cannot convert start_datetime '{start_datetime}' (type: {type(start_datetime)}) to datetime: {str(e)}"
            
            # Convert Excel number to Python float with better error handling
            try:
                minutes_to_add = round(float(jobtime))
            except (ValueError, TypeError) as e:
                return f"Error: Cannot convert jobtime '{jobtime}' (type: {type(jobtime)}) to float: {str(e)}"
            
            # Handle default data if not provided
            if calendar_rules_data is None:
                # Default calendar rules data
                calendar_rules_data = [
                    ['id', 'calendar_id', 'weekday', 'start_time', 'end_time'],
                    [1, 'default', 'Mon', '09:00', '17:00'],
                    [2, 'default', 'Tue', '09:00', '17:00'],
                    [3, 'default', 'Wed', '09:00', '17:00'],
                    [4, 'default', 'Thu', '09:00', '17:00'],
                    [5, 'default', 'Fri', '09:00', '17:00']
                ]
            
            if calendar_exceptions_data is None:
                # Default empty exceptions data
                calendar_exceptions_data = [
                    ['id', 'calendar_id', 'date', 'start_time', 'end_time', 'is_working']
                ]
            
            # Enhanced error handling for data conversion
            try:
                # Convert 2D arrays to pandas DataFrames
                if isinstance(calendar_rules_data, list) and len(calendar_rules_data) > 1:
                    rules_headers = calendar_rules_data[0]
                    rules_data = calendar_rules_data[1:]
                    rules_df = pd.DataFrame(rules_data, columns=rules_headers)
                else:
                    return f"Error: Invalid calendar rules data format. Type: {type(calendar_rules_data)}, Content: {str(calendar_rules_data)[:200]}"
                
                if isinstance(calendar_exceptions_data, list) and len(calendar_exceptions_data) >= 1:
                    exceptions_headers = calendar_exceptions_data[0]
                    exceptions_data = calendar_exceptions_data[1:] if len(calendar_exceptions_data) > 1 else []
                    exceptions_df = pd.DataFrame(exceptions_data, columns=exceptions_headers)
                else:
                    return f"Error: Invalid calendar exceptions data format. Type: {type(calendar_exceptions_data)}, Content: {str(calendar_exceptions_data)[:200]}"
            except Exception as e:
                return f"Error: DataFrame conversion failed: {str(e)}"
            
            # Validate DataFrames
            if rules_df.empty:
                return "Error: Calendar rules data is empty"
            
            # Load calendar data using existing functions
            rules = load_calendar_rules(rules_df)
            exceptions = load_calendar_exceptions(exceptions_df)
            
            # Check if calendar_id exists in rules
            if calendar_id not in rules:
                # Try to use the first available calendar ID
                if not rules:
                    return "Error: No calendar rules found in data"
                calendar_id = next(iter(rules.keys()))
            
            # Calculate date range for interval building (start date + reasonable buffer)
            start_date = start_dt.date()
            # Add buffer based on job size with conservative estimate
            # Assume 5 working hours per calendar day (accounts for weekends, holidays, etc.)
            # Add 50% safety margin and minimum 14 days for small jobs
            estimated_working_days = minutes_to_add / (5 * 60)  # 5 hours working time per calendar day
            buffer_days = max(14, int(estimated_working_days * 1.5) + 7)  # 50% margin + 7 day buffer
            end_date = start_date + timedelta(days=buffer_days)
            
            # Build working intervals
            working_intervals = build_working_intervals(
                rules, exceptions, calendar_id, start_date, end_date
            )
            
            # Calculate completion time
            completion_dt = add_working_minutes(start_dt, minutes_to_add, working_intervals)
            
            if completion_dt is None:
                return f"Error: Cannot complete {minutes_to_add} minutes of work within the calculated time range"
            
            # Convert datetime to Excel format
            if isinstance(completion_dt, datetime):
                return datetime_to_excel(completion_dt)
            else:
                return completion_dt
            
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
        start_datetime = datetime(2025, 1, 6, 9, 0)  # Monday 9 AM
        job_duration = 120  # 2 hours in minutes
        
        print(f"\nDemo: Adding {job_duration} minutes to {start_datetime}")
        print("-" * 50)
        
        for calendar_id in rules.keys():
            print(f"\nCalendar '{calendar_id}':")
            
            # Build working intervals for a reasonable range
            start_date = start_datetime.date()
            end_date = start_date + timedelta(days=30)
            
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