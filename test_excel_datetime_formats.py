import pandas as pd
from datetime import datetime
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from working_calendar import datetime_to_excel

def test_various_datetime_formats():
    """Test various datetime formats to understand Excel's behavior."""
    
    print("=== Testing Various Datetime Input Formats ===")
    
    # Test cases that might explain the 1970 vs 2025 issue
    test_cases = [
        # Standard formats
        ("String datetime", "2025-01-01 00:00:00"),
        ("String date only", "2025-01-01"),
        
        # Unix timestamp related
        ("Unix timestamp as int", 1735689600),  # 2025-01-01 00:00:00 UTC
        ("Unix timestamp as string", "1735689600"),
        
        # Excel date serial number
        ("Excel serial as int", 45658),  # 2025-01-01 in Excel
        ("Excel serial as float", 45658.0),
        ("Excel serial as string", "45658"),
        
        # Potential problem formats
        ("Days since epoch", (datetime(2025, 1, 1) - datetime(1970, 1, 1)).days),
        ("Seconds since epoch", int((datetime(2025, 1, 1) - datetime(1970, 1, 1)).total_seconds())),
    ]
    
    for name, value in test_cases:
        print(f"\n--- {name}: {value} (type: {type(value)}) ---")
        
        try:
            # Try to parse with pandas
            if isinstance(value, str):
                parsed = pd.to_datetime(value)
                if hasattr(parsed, 'to_pydatetime'):
                    py_dt = parsed.to_pydatetime()
                else:
                    py_dt = parsed
                print(f"  Pandas parsed as: {py_dt}")
                
                excel_val = datetime_to_excel(py_dt)
                print(f"  Excel value: {excel_val}")
                
                # Convert back to verify
                excel_epoch = datetime(1899, 12, 30)
                result_dt = excel_epoch + pd.Timedelta(days=excel_val)
                print(f"  Converts back to: {result_dt}")
                
            elif isinstance(value, (int, float)):
                # Try different interpretations
                
                # As Unix timestamp
                try:
                    unix_dt = pd.to_datetime(value, unit='s')
                    print(f"  As Unix timestamp: {unix_dt}")
                    if hasattr(unix_dt, 'to_pydatetime'):
                        unix_py = unix_dt.to_pydatetime()
                        excel_val = datetime_to_excel(unix_py)
                        print(f"  Excel value (Unix): {excel_val}")
                except:
                    print(f"  Cannot parse as Unix timestamp")
                
                # As Excel serial date
                try:
                    excel_epoch = datetime(1899, 12, 30)
                    excel_dt = excel_epoch + pd.Timedelta(days=value)
                    print(f"  As Excel serial: {excel_dt}")
                    excel_val = datetime_to_excel(excel_dt)
                    print(f"  Excel value (serial): {excel_val}")
                except:
                    print(f"  Cannot parse as Excel serial")
                    
                # As days since epoch
                try:
                    epoch_dt = datetime(1970, 1, 1) + pd.Timedelta(days=value)
                    print(f"  As days since 1970: {epoch_dt}")
                    excel_val = datetime_to_excel(epoch_dt)
                    print(f"  Excel value (epoch days): {excel_val}")
                except:
                    print(f"  Cannot parse as days since epoch")
                    
        except Exception as e:
            print(f"  ERROR: {str(e)}")

if __name__ == "__main__":
    test_various_datetime_formats()