import pandas as pd
from collections import deque, defaultdict
from pathlib import Path
from datetime import datetime, timedelta
import xlwings as xw
import sys
import os
# Add src directory to path when running as main script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))

try:
    from .config import get_file_path, get_data_path, get_project_root
except ImportError:
    from config import get_file_path, get_data_path, get_project_root

def explode_bom_iterative(independent_demand, items, bom):
    # Create lookup dictionaries
    lead_times = {item['item']: item['production_lead_time'] for item in items}
    bom_dict = defaultdict(list)
    for b in bom:
        bom_dict[b['parent_item']].append((b['child_item'], b['quantity_per']))
    
    total_demand = []
    queue = deque()
    
    # Initialize queue with independent demand
    for demand in independent_demand:
        # Convert due_date string to datetime object
        if isinstance(demand['due_date'], str):
            due_date_obj = datetime.strptime(demand['due_date'], '%Y-%m-%d')
        else:
            due_date_obj = demand['due_date']
        # Queue: (current_item, current_qty, current_due_date, parent_item, original_product_item, original_qty, original_due_date)
        queue.append((demand['item'], demand['quantity'], due_date_obj, None, demand['item'], demand['quantity'], due_date_obj))
    
    while queue:
        item, qty, due_date, parent_item, product_item, original_qty, original_due_date = queue.popleft()
        
        # Add current level to results
        if parent_item:  # Skip root level items
            total_demand.append({
                'product_item': product_item,
                'quantity': original_qty,
                'due_date': original_due_date.strftime('%Y-%m-%d') if isinstance(original_due_date, datetime) else original_due_date,
                'parent_item': parent_item,
                'child_item': item,
                'child_qty': qty,
                'child_due_date': due_date.strftime('%Y-%m-%d') if isinstance(due_date, datetime) else due_date
            })
        
        # Explode children
        for child_item, qty_per in bom_dict[item]:
            child_qty = qty * qty_per
            # Child due date = Parent due date - Parent's lead time
            # (Child must be ready before parent production starts)
            parent_lead_time_days = lead_times.get(item, 0)
            child_due = due_date - timedelta(days=parent_lead_time_days)
            queue.append((child_item, child_qty, child_due, item, product_item, original_qty, original_due_date))
    
    return total_demand

def bom_explosion_from_csv(dataset='current'):
    """
    Execute BOM explosion using CSV files from configured data directories.
    
    Args:
        dataset (str): Dataset name (current, simple, test, etc.) as defined in config.yaml
    
    Returns:
        str: Success message with output file path
    """
    try:
        # Get cross-platform file paths from configuration
        independent_demand_file = get_file_path(dataset, 'independent_demand')
        items_file = get_file_path(dataset, 'items')
        bom_file = get_file_path(dataset, 'bom')
        output_file = get_file_path(dataset, 'total_demand')
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Read input CSV files
        independent_demand = pd.read_csv(independent_demand_file).to_dict(orient='records')
        items = pd.read_csv(items_file).to_dict(orient='records')
        bom = pd.read_csv(bom_file).to_dict(orient='records')
        
        # Call the iterative explosion function
        total_demand = explode_bom_iterative(independent_demand, items, bom)
        
        # Convert result to DataFrame and write to CSV
        total_demand_df = pd.DataFrame(total_demand)
        total_demand_df.to_csv(output_file, index=False)

        return f'total_demand.csv generated successfully at {output_file}'
    
    except FileNotFoundError as e:
        return f'Error: Required input file not found - {e}'
    except Exception as e:
        return f'Error during BOM explosion: {e}'

def bom_explosion_from_excel():
    """
    Simple Excel wrapper for BOM explosion that reads data from Excel tables
    and writes results back to Excel.
    
    Expected Excel table structure in the current workbook:
    - Sheet 'independent_demand' with table containing: item, quantity, due_date
    - Sheet 'items' with table containing: item, production_lead_time
    - Sheet 'bom' with table containing: parent_item, child_item, quantity_per
    
    Output will be written to 'total_demand' sheet
    """
    # Get the active Excel workbook
    book = xw.Book.caller()
    
    # Read the tables using direct table reference
    independent_demand_df = book.sheets['independent_demand'].range('A1').options(pd.DataFrame, header=1, index=False, expand='table').value
    items_df = book.sheets['items'].range('A1').options(pd.DataFrame, header=1, index=False, expand='table').value
    bom_df = book.sheets['bom'].range('A1').options(pd.DataFrame, header=1, index=False, expand='table').value
    
    # Convert to dictionaries for the explosion function
    independent_demand = independent_demand_df.to_dict(orient='records')
    items = items_df.to_dict(orient='records')
    bom = bom_df.to_dict(orient='records')
    
    # Call the iterative explosion function
    total_demand = explode_bom_iterative(independent_demand, items, bom)
    
    # Convert result to DataFrame
    total_demand_df = pd.DataFrame(total_demand)
    
    # Write results to Excel
    if not total_demand_df.empty:
        # Create or get the total_demand sheet
        if 'total_demand' in [sheet.name for sheet in book.sheets]:
            output_sheet = book.sheets['total_demand']
            output_sheet.clear()
        else:
            output_sheet = book.sheets.add('total_demand', after = book.sheets['independent_demand'])
        
        # Write the DataFrame to Excel
        output_sheet.range('A1').options(pd.DataFrame, header=1, index=False).value = total_demand_df
        
        return f'BOM explosion completed successfully. {len(total_demand)} records written to total_demand sheet.'
    else:
        return 'BOM explosion completed but no demand records were generated.'

def main():
    """Main function to execute BOM explosion from CSV files."""
    # Usage with default dataset 'current'
    result_message = bom_explosion_from_csv()
    print(result_message)
    
    # Alternative usage with different datasets
    # result_message = bom_explosion_from_csv('simple')
    # result_message = bom_explosion_from_csv('test')
    # result_message = bom_explosion_from_csv('rnc')


if __name__ == "__main__":
    main()
