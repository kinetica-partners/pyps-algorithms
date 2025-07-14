from datetime import datetime, timedelta
import sys
sys.path.append('src')
from working_calendar import add_working_minutes, build_working_intervals, load_calendar_rules, load_calendar_exceptions
import pandas as pd

# Load the data
rules_df = pd.read_csv('data/calendar_rules.csv')
exceptions_df = pd.read_csv('data/calendar_exceptions.csv')
rules = load_calendar_rules(rules_df)
exceptions = load_calendar_exceptions(exceptions_df)

# Build just the first 8 days
start_date = datetime(2025, 1, 1).date()
end_date = datetime(2025, 1, 8).date()
working_intervals = build_working_intervals(rules, exceptions, 'calC', start_date, end_date)

print("Working intervals for first 8 days:")
for i, (start_dt, end_dt, cumu) in enumerate(working_intervals):
    duration = int((end_dt - start_dt).total_seconds() // 60)
    print(f"  {i}: {start_dt} to {end_dt} ({duration} min), cumulative: {cumu}")

print("\nTest: Add 4970 minutes from Jan 1 00:00")
start_test = datetime(2025, 1, 1, 0, 0)
result = add_working_minutes(start_test, 4970, working_intervals)
print(f"Result: {result}")

print("\nManual calculation:")
print("Target: 4970 minutes")
print("Cumulative at Day 7 start: 4260")
print("Minutes needed into Day 7: 4970 - 4260 = 710")
print("Day 7 start: 2025-01-07 09:00:00")
print("Expected: 2025-01-07 09:00:00 + 710 minutes = 2025-01-07 20:50:00")

# Let's test adding 710 minutes to the start of Day 7 directly
day7_start = datetime(2025, 1, 7, 9, 0)
direct_result = day7_start + timedelta(minutes=710)
print(f"Direct calculation: {day7_start} + 710 minutes = {direct_result}")
