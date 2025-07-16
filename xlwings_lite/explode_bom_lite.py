"""
xlwings lite BOM explosion module - Streamlined Core Implementation

This module is completely self-contained with no internal dependencies,
designed for xlwings Excel integration. Focuses on core BOM explosion
functionality with level tracking and natural fail-safe behavior.
"""

import pandas as pd
from collections import deque, defaultdict, namedtuple
from datetime import datetime, timedelta
from typing import List, Dict, Any
import xlwings as xw
from xlwings import script, func

# --- Core BOM Explosion Functions ---

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

# --- Excel Utility Functions ---

def excel_date_to_datetime(excel_date) -> datetime:
    """Convert Excel date to Python datetime."""
    if isinstance(excel_date, datetime):
        return excel_date
    elif isinstance(excel_date, str):
        try:
            return datetime.strptime(excel_date, '%Y-%m-%d')
        except ValueError:
            try:
                return datetime.strptime(excel_date, '%m/%d/%Y')
            except ValueError:
                return datetime.strptime(excel_date, '%d/%m/%Y')
    elif isinstance(excel_date, (int, float)):
        # Excel serial number
        excel_epoch = datetime(1899, 12, 30)
        return excel_epoch + timedelta(days=excel_date)
    else:
        raise ValueError(f"Cannot convert {excel_date} to datetime")

def datetime_to_excel_date(dt: datetime) -> str:
    """Convert Python datetime to Excel date string."""
    return dt.strftime('%Y-%m-%d')

def convert_excel_data_to_dict(excel_data: List[List], headers: List[str]) -> List[Dict]:
    """Convert Excel range data to list of dictionaries."""
    if not excel_data or len(excel_data) < 2:
        return []
    
    # Skip header row if it matches expected headers
    data_rows = excel_data[1:] if excel_data[0] == headers else excel_data
    
    result = []
    for row in data_rows:
        if len(row) >= len(headers):
            record = {}
            for i, header in enumerate(headers):
                record[header] = row[i]
            result.append(record)
    
    return result

# --- xlwings Functions ---

@func
def explode_bom_excel(
    independent_demand_data,
    items_data,
    bom_data
):
    """
    Excel function to perform BOM explosion.
    
    Args:
        independent_demand_data: 2D array with headers [item, quantity, due_date]
        items_data: 2D array with headers [item, production_lead_time]
        bom_data: 2D array with headers [parent_item, child_item, quantity_per]
        
    Returns:
        2D array with exploded BOM data including level field
    """
    try:
        # Convert to dictionaries
        independent_demand = convert_excel_data_to_dict(independent_demand_data, ['item', 'quantity', 'due_date'])
        items = convert_excel_data_to_dict(items_data, ['item', 'production_lead_time'])
        bom = convert_excel_data_to_dict(bom_data, ['parent_item', 'child_item', 'quantity_per'])
        
        # Perform explosion (natural fail-safe: empty BOM = only independent demand returned)
        result = explode_bom_iterative(independent_demand, items, bom)
        
        # Convert back to 2D array for Excel
        if not result:
            return [['No explosion results']]
        
        # Headers - include level field
        headers = ['product_item', 'quantity', 'due_date', 'parent_item', 'child_item', 'child_qty', 'child_due_date', 'level']
        output = [headers]
        
        # Data rows
        for record in result:
            row = [
                record.get('product_item', ''),
                record.get('quantity', 0),
                record.get('due_date', ''),
                record.get('parent_item', ''),
                record.get('child_item', ''),
                record.get('child_qty', 0),
                record.get('child_due_date', ''),
                record.get('level', 0)
            ]
            output.append(row)
        
        return output
        
    except Exception as e:
        return [['Error', str(e)]]

@func
def count_bom_levels(bom_data):
    """
    Excel function to count BOM levels.
    
    Returns:
        Number of BOM levels
    """
    try:
        # Convert to dictionaries
        bom = convert_excel_data_to_dict(bom_data, ['parent_item', 'child_item', 'quantity_per'])
        
        if not bom:
            return 0
        
        # Build parent-child relationships
        children = defaultdict(set)
        all_items = set()
        
        for record in bom:
            parent = record['parent_item']
            child = record['child_item']
            children[parent].add(child)
            all_items.add(parent)
            all_items.add(child)
        
        # Find top-level items (items that are not children of any other item)
        all_parents = set(children.keys())
        all_children = set()
        for child_set in children.values():
            all_children.update(child_set)
        
        top_level_items = all_parents - all_children
        
        if not top_level_items:
            return 0
        
        # Calculate maximum depth
        max_depth = 0
        
        def get_depth(item, visited=None):
            if visited is None:
                visited = set()
            
            if item in visited:
                return 0  # Circular reference
            
            visited.add(item)
            
            if item not in children:
                return 1
            
            max_child_depth = 0
            for child in children[item]:
                child_depth = get_depth(child, visited.copy())
                max_child_depth = max(max_child_depth, child_depth)
            
            return max_child_depth + 1
        
        for top_item in top_level_items:
            depth = get_depth(top_item)
            max_depth = max(max_depth, depth)
        
        return max_depth
        
    except Exception as e:
        return f"Error: {str(e)}"

# --- Excel Script Functions ---

@script
def bom_explosion_from_excel(book: xw.Book):
    """
    Excel script to perform BOM explosion from Excel tables.
    
    Expected Excel table structure in the current workbook:
    - Sheet 'independent_demand' with table containing: item, quantity, due_date
    - Sheet 'items' with table containing: item, production_lead_time
    - Sheet 'bom' with table containing: parent_item, child_item, quantity_per
    
    Output will be written to 'total_demand' sheet
    """
    try:
        # Read the tables using direct table reference
        independent_demand_df = book.sheets['independent_demand'].range('A10').options(pd.DataFrame, header=1, index=False, expand='table').value
        items_df = book.sheets['items'].range('A10').options(pd.DataFrame, header=1, index=False, expand='table').value
        bom_df = book.sheets['bom'].range('A10').options(pd.DataFrame, header=1, index=False, expand='table').value
        
        # Convert to dictionaries for the explosion function
        independent_demand = independent_demand_df.to_dict(orient='records')
        items = items_df.to_dict(orient='records')
        bom = bom_df.to_dict(orient='records')
        
        # Call the iterative explosion function (natural fail-safe)
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
                output_sheet = book.sheets.add('total_demand', after=book.sheets['independent_demand'])
                            
            # Write the DataFrame to Excel
            output_sheet.range('A10').options(pd.DataFrame, header=1, index=False).value = total_demand_df
            book.sheets['total_demand'].range('A10:H10').font.bold = True
            book.sheets['total_demand'].range('A10:H10').autofit()

            return f'BOM explosion completed successfully. {len(total_demand)} records written to total_demand sheet.'
        else:
            return 'No explosion results generated.'
            
    except Exception as e:
        return f"Error: {str(e)}"