import pandas as pd
from collections import deque, defaultdict, namedtuple
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
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

# Constants
DATE_FORMAT = '%Y-%m-%d'
DEFAULT_LEAD_TIME = 0
INDEPENDENT_DEMAND_LEVEL = 0
LEVEL_INCREMENT = 1

def _format_date(date_obj) -> str:
    """Helper function to format date objects consistently."""
    return date_obj.strftime(DATE_FORMAT) if isinstance(date_obj, datetime) else date_obj

def _parse_date_string(date_input) -> datetime:
    """Helper function to parse date strings into datetime objects."""
    if isinstance(date_input, str):
        try:
            return datetime.strptime(date_input, DATE_FORMAT)
        except ValueError as e:
            raise ValueError(f"Invalid date format '{date_input}'. Expected format: {DATE_FORMAT}") from e
    return date_input

def _is_valid_demand(demand) -> bool:
    """Helper function to validate demand record has required fields."""
    return (demand.get('item') and
            demand.get('quantity') and
            demand.get('due_date'))

# Named tuple for queue items to improve readability and reduce errors
QueueItem = namedtuple('QueueItem', [
    'item',           # current item being processed
    'qty',            # quantity of current item
    'due_date',       # due date for current item
    'parent_item',    # parent item (None for level 0)
    'product_item',   # original product item
    'original_qty',   # original quantity from demand
    'original_due_date',  # original due date from demand
    'level',          # BOM level (0 = independent demand)
    'parent_due_date' # parent item's due date
])

def explode_bom_iterative(independent_demand: List[Dict], items: List[Dict], bom: List[Dict]) -> List[Dict]:
    """
    Iterative BOM explosion algorithm.
    
    Args:
        independent_demand: List of demand records with item, quantity, due_date
        items: List of item records with item, production_lead_time
        bom: List of BOM records with parent_item, child_item, quantity_per
        
    Returns:
        List of exploded demand records
    """
    # Create lookup dictionaries
    lead_times = {item['item']: item['production_lead_time'] for item in items if item.get('item')}
    bom_dict = defaultdict(list)
    for bom_item in bom:
        if bom_item.get('parent_item') and bom_item.get('child_item'):
            bom_dict[bom_item['parent_item']].append((bom_item['child_item'], bom_item['quantity_per']))
    
    total_demand = []
    
    # Process each independent demand item completely before moving to next
    for demand in independent_demand:
        # Skip empty/invalid demand records
        if not _is_valid_demand(demand):
            continue
        
        # Convert due_date string to datetime object
        due_date_obj = _parse_date_string(demand['due_date'])
        
        # Create separate queue for this independent demand item
        queue = deque()
        # Add initial demand item to queue
        queue.append(QueueItem(
            item=demand['item'],
            qty=demand['quantity'],
            due_date=due_date_obj,
            parent_item=None,
            product_item=demand['item'],
            original_qty=demand['quantity'],
            original_due_date=due_date_obj,
            level=INDEPENDENT_DEMAND_LEVEL,
            parent_due_date=due_date_obj
        ))
        
        # Process this independent demand item completely
        while queue:
            queue_item = queue.popleft()
            
            # Add current level to results - ALWAYS add every item including independent demand (level 0)
            total_demand.append({
                'product_item': queue_item.product_item,
                'quantity': queue_item.original_qty,
                'due_date': _format_date(queue_item.original_due_date if queue_item.parent_item is None else queue_item.parent_due_date),
                'parent_item': queue_item.item if queue_item.parent_item is None else queue_item.parent_item,
                'child_item': queue_item.item,
                'child_qty': queue_item.qty,
                'child_due_date': _format_date(queue_item.due_date),
                'level': queue_item.level
            })
            
            # Explode children
            for child_item, qty_per in bom_dict[queue_item.item]:
                child_qty = queue_item.qty * qty_per
                # Child due date = Parent due date - Parent's lead time
                # (Child must be ready before parent production starts)
                parent_lead_time_days = lead_times.get(queue_item.item, DEFAULT_LEAD_TIME)
                child_due = queue_item.due_date - timedelta(days=parent_lead_time_days)
                # Pass the current item's due_date as the parent_due_date for children
                queue.append(QueueItem(
                    item=child_item,
                    qty=child_qty,
                    due_date=child_due,
                    parent_item=queue_item.item,
                    product_item=queue_item.product_item,
                    original_qty=queue_item.original_qty,
                    original_due_date=queue_item.original_due_date,
                    level=queue_item.level + LEVEL_INCREMENT,
                    parent_due_date=queue_item.due_date
                ))
    
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
