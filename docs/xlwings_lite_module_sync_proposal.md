# xlwings Lite Modules - Comprehensive Phased Proposal

## Executive Summary

Create a comprehensive xlwings lite integration strategy that balances immediate functionality with long-term maintainability. This proposal outlines a phased approach from manual duplication to automated generation, ensuring DRY principles while maintaining xlwings lite compatibility.

## Current Problem Analysis

### Issues with Current Implementation:
1. **Config Dependency**: Current modules use `get_file_path()` and `get_data_path()` from config system
2. **Import Complexity**: xlwings lite functions can't handle complex imports from the config system
3. **Path Resolution**: xlwings lite needs hardcoded or self-contained path resolution
4. **Portability**: Current modules are tied to the project structure

### What Worked Before:
- **Self-contained functions**: No external config dependencies
- **Hardcoded defaults**: Built-in calendar rules and exception handling
- **Direct xlwings integration**: Simple `@func` and `@script` decorators
- **Excel compatibility**: Proper Excel date format handling

## Comprehensive Phased Architecture

### **Phase 1: Manual Duplication (Immediate Implementation)**
**Goal**: Get xlwings lite working quickly with manual code duplication
**Timeline**: 1-2 weeks

```
src/
├── xlwings_lite/
│   ├── __init__.py
│   ├── working_calendar_lite.py    # FULLY SELF-CONTAINED (manual duplication)
│   └── explode_bom_lite.py         # FULLY SELF-CONTAINED (manual duplication)
tests/
├── test_xlwings_consistency.py     # NEW: Consistency testing
```

**Characteristics**:
- **Manual copy-paste** of core logic into lite modules
- **Self-contained modules** with no internal dependencies
- **Immediate xlwings lite compatibility**
- **Consistency tests** to ensure lite modules match main modules

### **Phase 2: Core Module Extraction (DRY Foundation)**
**Goal**: Extract shared logic to eliminate duplication
**Timeline**: 2-3 weeks after Phase 1

```
src/
├── core/
│   ├── __init__.py
│   ├── working_calendar_core.py    # Pure algorithms, no config/xlwings
│   ├── explode_bom_core.py         # Pure algorithms, no config/xlwings
│   └── utilities_core.py           # Shared utilities
├── working_calendar.py             # Uses core + config system
├── explode_bom.py                  # Uses core + config system
├── xlwings_lite/
│   ├── working_calendar_lite.py    # Uses core + embedded defaults
│   └── explode_bom_lite.py         # Uses core + embedded defaults
tests/
├── test_core_algorithms.py         # Test core logic
├── test_xlwings_consistency.py     # Consistency testing
```

**Characteristics**:
- **Core modules** contain pure algorithms with no external dependencies
- **Main modules** use core + config system
- **Lite modules** use core + embedded defaults (still self-contained for xlwings)
- **Shared algorithm testing** ensures consistency across all implementations

### **Phase 3: Automated Generation (Full DRY)**
**Goal**: Script-generated lite modules for ultimate maintainability
**Timeline**: 3-4 weeks after Phase 2

```
src/
├── core/                           # Pure algorithms
├── working_calendar.py             # Main implementation
├── explode_bom.py                  # Main implementation
├── xlwings_lite/
│   ├── working_calendar_lite.py    # GENERATED from core + template
│   └── explode_bom_lite.py         # GENERATED from core + template
scripts/
├── generate_xlwings_lite.py        # Generation script
├── templates/
│   ├── working_calendar_template.py
│   └── explode_bom_template.py
tests/
├── test_generated_consistency.py   # Test generated vs main modules
```

**Characteristics**:
- **Generated lite modules** from core algorithms + templates
- **Template-based approach** for xlwings decorators and Excel handling
- **Automated consistency** through generation process
- **Single source of truth** for algorithms

## Detailed Phase 1 Implementation

### **Phase 1A: Manual Duplication Strategy**

#### **working_calendar_lite.py Structure**
```python
"""
xlwings Lite Working Calendar Module
MANUALLY DUPLICATED from main working_calendar.py
Self-contained with no external dependencies
"""

import pandas as pd
from datetime import datetime, timedelta, time, date
import xlwings as xw
from xlwings import func, arg, script

# --- EMBEDDED EXCEL UTILITIES ---
def datetime_to_excel(dt):
    """Convert Python datetime to Excel date serial number."""
    excel_epoch = datetime(1899, 12, 30)
    delta = dt - excel_epoch
    return float(delta.days) + (delta.seconds + delta.microseconds / 1e6) / 86400

def excel_time_to_string(excel_time):
    """Convert Excel time serial number to HH:MM format."""
    if isinstance(excel_time, (int, float)):
        total_minutes = round(excel_time * 24 * 60)
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours:02d}:{minutes:02d}"
    return str(excel_time)

def excel_boolean_to_python(excel_bool):
    """Convert Excel boolean value to Python boolean."""
    if isinstance(excel_bool, bool):
        return excel_bool
    elif isinstance(excel_bool, (int, float)):
        return bool(excel_bool)
    elif isinstance(excel_bool, str):
        return excel_bool.lower().strip() in ('true', '1', 'yes', 'on')
    return False

# --- EMBEDDED WORKING CALENDAR CORE LOGIC ---
# (Complete copy from docs/xlwings_last_working_versions.md)
def load_calendar_rules(rules_dataframe):
    """Load calendar rules from dataframe into structured format."""
    # ... complete implementation

def load_calendar_exceptions(exceptions_dataframe):
    """Load calendar exceptions from dataframe into structured format."""
    # ... complete implementation

def build_working_intervals(rules, exceptions, calendar_id, start_date, end_date):
    """Build working intervals for a date range."""
    # ... complete implementation

def add_working_minutes(start_dt, minutes_to_add, working_intervals):
    """Add working minutes using precomputed working intervals."""
    # ... complete implementation

# --- EMBEDDED DEFAULT DATA ---
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

# --- XLWINGS LITE FUNCTIONS ---
@func
def calculate_working_completion_time(
    start_datetime,
    jobtime,
    calendar_rules_data=None,
    calendar_exceptions_data=None,
    calendar_id="default"
):
    """Calculate working time completion datetime for Excel."""
    # ... complete implementation using embedded functions

@func
def test_working_calendar():
    """Simple test function returning known Excel date."""
    test_date = datetime(2025, 1, 1, 10, 10, 0)
    return datetime_to_excel(test_date)

@func
def greet():
    """Basic xlwings test function."""
    return "Hello from xlwings lite working calendar!"
```

### **Phase 1B: AST Consistency Testing ✅ IMPLEMENTED**

**Problem Solved**: xlwings lite modules cannot be imported due to decorator and import conflicts.

**Solution**: AST (Abstract Syntax Tree) comparison for pure algorithmic validation.

#### **Key Benefits of AST Approach**:
- ✅ **No runtime imports** - Pure static analysis, no xlwings dependency
- ✅ **Language-aware** - Understands Python syntax vs simple text comparison
- ✅ **Ignores decorators** - `@xw.func` decorators completely ignored
- ✅ **Detects real differences** - Catches algorithmic changes, not cosmetic ones
- ✅ **Comprehensive reporting** - Shows exactly which functions match/differ

#### **Implementation Files**:
- `tests/helpers/ast_comparison.py` - AST comparison engine (173 lines)
- `tests/test_ast_consistency.py` - Pytest integration for consistency testing
- `tests/helpers/__init__.py` - Package structure for helpers
- `docs/AST_COMPARISON_DEMO.md` - Documentation and examples

#### **Working Demo Results**:
```
AST COMPARISON REPORT
================================================================================
✅ Identical Functions (2): ['calculate_working_hours', 'excel_time_to_string']
⚠️  Different Functions (2): ['load_calendar_data', 'main']
📊 Summary: 2/4 functions identical (50.0%)
================================================================================
```

#### **tests/test_ast_consistency.py** (Updated Implementation)
```python
"""
Pytest test for AST consistency between main and xlwings lite modules.
This test ensures that the core algorithmic logic remains identical.
"""

import pytest
from pathlib import Path
import sys

# Add tests/helpers to path for imports
sys.path.append(str(Path(__file__).parent / "helpers"))

from ast_comparison import compare_modules, print_comparison_report

class TestASTConsistency:
    """Test AST consistency between main and xlwings lite modules."""
    
    def test_working_calendar_consistency_when_available(self):
        """Test consistency for working_calendar modules when xlwings lite is available."""
        main_file = Path("src/working_calendar.py")
        lite_file = Path("xlwings_lite/working_calendar_lite.py")
        
        # Skip if either file doesn't exist
        if not main_file.exists() or not lite_file.exists():
            pytest.skip("Working calendar files not found")
        
        results = compare_modules(main_file, lite_file, normalize_names=False)
        print_comparison_report(results)
        
        # Define core algorithmic functions that MUST be identical
        core_functions = {
            'excel_time_to_string',
            'excel_boolean_to_python',
            'parse_time_string',
            'calculate_working_hours',
            'is_working_day'
        }
        
        # Check which core functions exist and are identical
        existing_core_functions = core_functions & set(results['comparisons'].keys())
        identical_core_functions = existing_core_functions & set(results['identical'])
        
        # All existing core functions should be identical
        assert identical_core_functions == existing_core_functions, \
            f"Core functions should be identical. Differences found in: {existing_core_functions - identical_core_functions}"
        # Report coverage of core functions
        coverage = len(identical_core_functions) / len(core_functions) if core_functions else 0
        print(f"\nCore function consistency: {len(identical_core_functions)}/{len(core_functions)} ({coverage*100:.1f}%)")
        
        # Must have at least 80% core function consistency
        assert coverage >= 0.8, f"Core function consistency too low: {coverage*100:.1f}%"
    def test_explode_bom_consistency_when_available(self):
        """Test consistency for explode_bom modules when xlwings lite is available."""
        main_file = Path("src/explode_bom.py")
        lite_file = Path("xlwings_lite/explode_bom_lite.py")
        
        # Skip if either file doesn't exist
        if not main_file.exists() or not lite_file.exists():
            pytest.skip("Explode BOM files not found")
        
        results = compare_modules(main_file, lite_file, normalize_names=False)
        print_comparison_report(results)
        
        # Define core algorithmic functions for BOM explosion
        core_functions = {
            'explode_bom',
            'explode_bom_iterative',
            'calculate_total_demand',
            'process_bom_level'
        }
        
        # Check consistency
        existing_core_functions = core_functions & set(results['comparisons'].keys())
        identical_core_functions = existing_core_functions & set(results['identical'])
        
        # All existing core functions should be identical
        assert identical_core_functions == existing_core_functions, \
            f"Core BOM functions should be identical. Differences found in: {existing_core_functions - identical_core_functions}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
```

## Phase 2: Core Module Extraction

### **Phase 2A: Core Algorithm Modules**

#### **src/core/working_calendar_core.py**
```python
"""
Core working calendar algorithms.
Pure functions with no external dependencies.
Used by both main and lite modules.
"""

from datetime import datetime, timedelta, time, date
from typing import List, Dict, Tuple, Optional
from bisect import bisect_right

def load_calendar_rules(rules_dataframe) -> Dict[str, Dict[int, List[Tuple[time, time]]]]:
    """Load calendar rules from dataframe into structured format."""
    # ... pure implementation with no config dependencies

def load_calendar_exceptions(exceptions_dataframe) -> Dict[str, Dict[str, List[Tuple[time, time, bool]]]]:
    """Load calendar exceptions from dataframe into structured format."""
    # ... pure implementation

def build_working_intervals(
    rules: Dict[str, Dict[int, List[Tuple[time, time]]]],
    exceptions: Dict[str, Dict[str, List[Tuple[time, time, bool]]]],
    calendar_id: str,
    start_date: date,
    end_date: date
) -> List[Tuple[datetime, datetime, int]]:
    """Build working intervals for a date range."""
    # ... pure implementation

def add_working_minutes(
    start_dt: datetime,
    minutes_to_add: int,
    working_intervals: List[Tuple[datetime, datetime, int]]
) -> Optional[datetime]:
    """Add working minutes using precomputed working intervals."""
    # ... pure implementation
```

#### **src/core/utilities_core.py**
```python
"""
Core utility functions for Excel integration.
Pure functions with no external dependencies.
"""

from datetime import datetime, timedelta, time

def datetime_to_excel(dt: datetime) -> float:
    """Convert Python datetime to Excel date serial number."""
    excel_epoch = datetime(1899, 12, 30)
    delta = dt - excel_epoch
    return float(delta.days) + (delta.seconds + delta.microseconds / 1e6) / 86400

def excel_to_datetime(excel_serial: float) -> datetime:
    """Convert Excel serial number to Python datetime."""
    if excel_serial > 20000:  # Modern dates
        excel_epoch = datetime(1899, 12, 30)
        return excel_epoch + timedelta(days=excel_serial)
    else:
        raise ValueError(f"Invalid Excel serial number: {excel_serial}")

def excel_time_to_string(excel_time) -> str:
    """Convert Excel time serial number to HH:MM format."""
    if isinstance(excel_time, (int, float)):
        total_minutes = round(excel_time * 24 * 60)
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours:02d}:{minutes:02d}"
    return str(excel_time)

def excel_boolean_to_python(excel_bool) -> bool:
    """Convert Excel boolean value to Python boolean."""
    if isinstance(excel_bool, bool):
        return excel_bool
    elif isinstance(excel_bool, (int, float)):
        return bool(excel_bool)
    elif isinstance(excel_bool, str):
        return excel_bool.lower().strip() in ('true', '1', 'yes', 'on')
    return False
```

### **Phase 2B: Updated Module Structure**

#### **src/working_calendar.py** (Updated)
```python
"""
Main working calendar module using config system.
Uses core algorithms + config for data loading.
"""

import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Import from core
from .core.working_calendar_core import (
    load_calendar_rules,
    load_calendar_exceptions,
    build_working_intervals,
    add_working_minutes
)

# Import config system
from .config import get_file_path, get_data_path

def calculate_working_completion_time(
    start_datetime: datetime,
    jobtime: int,
    dataset: str = "current",
    calendar_id: str = "default"
) -> datetime:
    """Calculate working completion time using config system."""
    # Load data using config system
    rules_file = get_file_path(dataset, 'calendar_rules')
    exceptions_file = get_file_path(dataset, 'calendar_exceptions')
    
    rules_df = pd.read_csv(rules_file)
    exceptions_df = pd.read_csv(exceptions_file)
    
    # Use core algorithms
    rules = load_calendar_rules(rules_df)
    exceptions = load_calendar_exceptions(exceptions_df)
    
    # ... rest of implementation using core functions
```

#### **src/xlwings_lite/working_calendar_lite.py** (Updated)
```python
"""
xlwings Lite working calendar module.
Uses core algorithms + embedded defaults.
Self-contained for xlwings compatibility.
"""

import pandas as pd
from datetime import datetime, timedelta, time, date
import xlwings as xw
from xlwings import func, arg, script

# EMBEDDED CORE ALGORITHMS (still needed for xlwings lite compatibility)
# Copy from core but embed directly to avoid import issues
def load_calendar_rules(rules_dataframe):
    """Load calendar rules from dataframe into structured format."""
    # ... exact copy from core module

def load_calendar_exceptions(exceptions_dataframe):
    """Load calendar exceptions from dataframe into structured format."""
    # ... exact copy from core module

# ... rest of embedded core functions

@func
def calculate_working_completion_time(...):
    """xlwings lite function using embedded core algorithms."""
    # ... implementation using embedded core functions
```

## Phase 3: Automated Generation

### **Phase 3A: Generation Script**

#### **scripts/generate_xlwings_lite.py**
```python
"""
Generate xlwings lite modules from core algorithms and templates.
Ensures perfect consistency between main and lite modules.
"""

import os
import ast
from pathlib import Path
from typing import Dict, List

class XlwingsLiteGenerator:
    """Generate xlwings lite modules from core and templates."""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.core_path = self.project_root / "src" / "core"
        self.templates_path = self.project_root / "scripts" / "templates"
        self.output_path = self.project_root / "src" / "xlwings_lite"
    
    def extract_core_functions(self, core_module: str) -> Dict[str, str]:
        """Extract function definitions from core module."""
        core_file = self.core_path / f"{core_module}.py"
        
        with open(core_file, 'r') as f:
            tree = ast.parse(f.read())
        
        functions = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract function source code
                function_lines = ast.get_source_segment(f.read(), node).split('\n')
                functions[node.name] = '\n'.join(function_lines)
        
        return functions
    
    def generate_from_template(self, template_name: str, core_functions: Dict[str, str]) -> str:
        """Generate xlwings lite module from template and core functions."""
        template_file = self.templates_path / f"{template_name}.py"
        
        with open(template_file, 'r') as f:
            template_content = f.read()
        
        # Replace placeholders with core functions
        for func_name, func_code in core_functions.items():
            placeholder = f"# EMBED_{func_name.upper()}"
            template_content = template_content.replace(placeholder, func_code)
        
        return template_content
    
    def generate_working_calendar_lite(self):
        """Generate working_calendar_lite.py from core and template."""
        # Extract core functions
        core_functions = self.extract_core_functions("working_calendar_core")
        utility_functions = self.extract_core_functions("utilities_core")
        
        # Combine all functions
        all_functions = {**core_functions, **utility_functions}
        
        # Generate from template
        generated_content = self.generate_from_template("working_calendar_template", all_functions)
        
        # Write output
        output_file = self.output_path / "working_calendar_lite.py"
        with open(output_file, 'w') as f:
            f.write(generated_content)
        
        print(f"Generated: {output_file}")
    
    def generate_all(self):
        """Generate all xlwings lite modules."""
        self.generate_working_calendar_lite()
        # ... generate other modules

if __name__ == "__main__":
    generator = XlwingsLiteGenerator(".")
    generator.generate_all()
```

### **Phase 3B: Templates**

#### **scripts/templates/working_calendar_template.py**
```python
"""
xlwings Lite Working Calendar Module - GENERATED
DO NOT EDIT DIRECTLY - Generated from core modules and template

Generated on: {generation_date}
From core modules: working_calendar_core.py, utilities_core.py
"""

import pandas as pd
from datetime import datetime, timedelta, time, date
import xlwings as xw
from xlwings import func, arg, script

# --- EMBEDDED CORE UTILITIES ---
# EMBED_DATETIME_TO_EXCEL
# EMBED_EXCEL_TIME_TO_STRING
# EMBED_EXCEL_BOOLEAN_TO_PYTHON

# --- EMBEDDED CORE ALGORITHMS ---
# EMBED_LOAD_CALENDAR_RULES
# EMBED_LOAD_CALENDAR_EXCEPTIONS
# EMBED_BUILD_WORKING_INTERVALS
# EMBED_ADD_WORKING_MINUTES

# --- EMBEDDED DEFAULT DATA ---
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

# --- XLWINGS LITE FUNCTIONS ---
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
        # Handle default data
        if calendar_rules_data is None:
            calendar_rules_data = get_default_calendar_rules()
        if calendar_exceptions_data is None:
            calendar_exceptions_data = get_default_calendar_exceptions()
        
        # Use embedded core functions
        # ... implementation using embedded functions
        
        return datetime_to_excel(completion_dt)
    except Exception as e:
        return f"Error: {str(e)}"

@func
def test_working_calendar():
    """Simple test function returning known Excel date."""
    test_date = datetime(2025, 1, 1, 10, 10, 0)
    return datetime_to_excel(test_date)

@func
def greet():
    """Basic xlwings test function."""
    return "Hello from xlwings lite working calendar! (Generated)"
```

## Comprehensive Testing Strategy

### **Consistency Testing Framework**
```python
"""
Comprehensive consistency testing framework.
Tests all phases for algorithmic correctness.
"""

class ConsistencyTester:
    """Test consistency across all implementations."""
    
    def test_phase1_manual_consistency(self):
        """Test Phase 1 manual duplication consistency."""
        # Test main vs lite (manual)
        
    def test_phase2_core_consistency(self):
        """Test Phase 2 core extraction consistency."""
        # Test main vs lite (using core)
        
    def test_phase3_generated_consistency(self):
        """Test Phase 3 generated modules consistency."""
        # Test main vs lite (generated)
        
    def test_cross_phase_consistency(self):
        """Test consistency across all phases."""
        # Test Phase 1 vs Phase 2 vs Phase 3
        
    def generate_comprehensive_report(self):
        """Generate comprehensive consistency report."""
        # ... detailed reporting
```

## Implementation Timeline

### **Phase 1: Immediate (1-2 weeks)**
- [x] Create manual working_calendar_lite.py
- [x] Create manual explode_bom_lite.py
- [x] Implement consistency testing
- [x] Test with Excel integration

### **Phase 2: DRY Foundation (2-3 weeks)**
- [ ] Extract core algorithms to src/core/
- [ ] Update main modules to use core
- [ ] Update lite modules to use core (but still embedded)
- [ ] Enhance consistency testing

### **Phase 3: Automation (3-4 weeks)**
- [ ] Create generation script
- [ ] Create templates
- [ ] Generate lite modules automatically
- [ ] Implement cross-phase consistency testing

## Benefits of Phased Approach

### **Immediate Benefits (Phase 1)**
- **Working xlwings integration** within days
- **Manual control** over lite modules
- **Proven consistency** through testing
- **No complex automation** required

### **Medium-term Benefits (Phase 2)**
- **Reduced duplication** through core modules
- **Easier maintenance** of core algorithms
- **Better testing** of pure algorithms
- **Foundation for automation**

### **Long-term Benefits (Phase 3)**
- **Perfect consistency** through generation
- **Minimal maintenance** overhead
- **Single source of truth** for algorithms
- **Automated testing** and validation

## Risk Mitigation

### **Phase 1 Risks**
- **Manual sync burden**: Mitigated by comprehensive consistency testing
- **Code duplication**: Accepted as temporary trade-off for quick implementation

### **Phase 2 Risks**
- **xlwings import issues**: Mitigated by keeping lite modules self-contained
- **Complexity increase**: Mitigated by clear separation of concerns

### **Phase 3 Risks**
- **Generation complexity**: Mitigated by simple template-based approach
- **Debugging difficulty**: Mitigated by clear generation process and source mapping

## Conclusion

This comprehensive phased approach provides:

1. **Immediate xlwings lite functionality** (Phase 1)
2. **DRY foundation** with core modules (Phase 2)
3. **Automated generation** for long-term maintainability (Phase 3)
4. **Consistency testing** throughout all phases
5. **Clear migration path** from manual to automated

The approach balances quick implementation with long-term maintainability, ensuring that xlwings lite integration works immediately while building toward a fully automated, DRY solution.