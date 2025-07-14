#!/usr/bin/env python3
"""
PyPS Scheduling Algorithms - Main Entry Point

This is a simplified, portable repository containing two main scheduling algorithms:
1. BOM Explosion - Explodes Bill of Materials with lead time calculations
2. Working Calendar - Calculates working time completion with calendar rules and exceptions

Both modules read their input data from ./data/current/ directory.
"""

import sys
import os
from datetime import datetime


def main():
    """Main entry point for PyPS Scheduling Algorithms."""
    print("PyPS Scheduling Algorithms")
    print("=" * 50)
    print()
    
    # Check if data directory exists
    data_dir = "./data/current"
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory '{data_dir}' not found!")
        print("Please ensure the following files exist:")
        print("  - ./data/current/bom.csv")
        print("  - ./data/current/items.csv") 
        print("  - ./data/current/independent_demand.csv")
        print("  - ./data/current/calendar_rules.csv")
        print("  - ./data/current/calendar_exceptions.csv")
        return 1
    
    print("Available algorithms:")
    print("1. BOM Explosion (Bill of Materials)")
    print("2. Working Calendar (Working Time Calculations)")
    print("3. Run both")
    print()
    
    choice = input("Select algorithm (1, 2, 3, or 'q' to quit): ").strip()
    
    if choice.lower() == 'q':
        print("Goodbye!")
        return 0
    
    if choice == '1' or choice == '3':
        print("\n" + "=" * 50)
        print("Running BOM Explosion...")
        try:
            # Import and run BOM explosion
            sys.path.insert(0, 'src')
            from explode_bom import main as bom_main
            bom_main()
        except Exception as e:
            print(f"Error running BOM explosion: {e}")
    
    if choice == '2' or choice == '3':
        print("\n" + "=" * 50)
        print("Running Working Calendar Demo...")
        try:
            # Import and run working calendar
            sys.path.insert(0, 'src')
            from working_calendar import main as calendar_main
            calendar_main()
        except Exception as e:
            print(f"Error running working calendar: {e}")
    
    if choice not in ['1', '2', '3']:
        print("Invalid choice. Please run again and select 1, 2, or 3.")
        return 1
    
    print("\n" + "=" * 50)
    print("PyPS Scheduling Algorithms completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
