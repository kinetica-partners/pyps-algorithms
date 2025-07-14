# PyPS Scheduling Algorithms

A simplified, portable repository containing two core scheduling algorithms for production planning and scheduling with cross-platform path management.

## Overview

This repository provides two main scheduling modules:

1. **BOM Explosion** - Explodes Bill of Materials with lead time calculations
2. **Working Calendar** - Calculates working time completion with calendar rules and exceptions

Both modules are designed to be fully portable with cross-platform compatible file paths that work identically on Windows, macOS, and Linux. The system uses YAML-based configuration for centralized path management.

## Project Structure

```
PyPS_Scheduling_Algorithms/
├── src/
│   ├── config.py               # Portable path configuration module
│   ├── explode_bom.py          # BOM explosion with main()
│   └── working_calendar.py     # Working calendar with main()
├── tests/
│   ├── test_explode_bom.py     # BOM explosion tests
│   ├── test_working_calendar.py # Working calendar tests
│   └── test_data/
│       ├── baseline.json       # Test baseline data
│       └── refactor_baseline.json
├── data/
│   └── current/                # Primary input data directory
│       ├── bom.csv
│       ├── items.csv
│       ├── independent_demand.csv
│       ├── calendar_rules.csv
│       └── calendar_exceptions.csv
├── docs/                       # Documentation
├── config.yaml                 # Portable path configuration
├── main.py                     # Main entry point
├── pyproject.toml             # Project configuration
└── README.md
```

## Installation

1. Ensure you have Python 3.12+ installed
2. Install dependencies:
   ```bash
   pip install pandas pytest xlwings pyyaml
   ```

## Portable Configuration System

This project uses a YAML-based configuration system for cross-platform compatibility:

- **[`config.yaml`](config.yaml)** - Centralized path configuration
- **[`src/config.py`](src/config.py)** - Configuration loader with pathlib.Path support
- **Automatic root detection** - Finds project root regardless of execution directory
- **Cross-platform paths** - Works identically on Windows, macOS, and Linux

The configuration system automatically:
- Detects the project root directory
- Resolves all paths relative to project root
- Uses forward slashes in config, converts to OS-specific paths
- Supports multiple datasets (current, test, baseline)

## Usage

### Option 1: Interactive Main Script
```bash
python main.py
```
This provides an interactive menu to run either algorithm or both.

### Option 2: Run Individual Modules
```bash
# BOM Explosion (uses current dataset by default)
python src/explode_bom.py

# Working Calendar Demo
python src/working_calendar.py
```

### Option 3: Import as Modules (Portable)
```python
from src.explode_bom import bom_explosion_from_csv
from src.working_calendar import calculate_working_completion_time

# BOM explosion with portable configuration
result = bom_explosion_from_csv(dataset='current')

# Or specify different dataset
result = bom_explosion_from_csv(dataset='test')

# Working calendar calculation (uses portable paths automatically)
completion_time = calculate_working_completion_time(
    start_datetime=datetime(2025, 1, 6, 9, 0),
    jobtime=120,  # minutes
    calendar_id="default"
)
```

### Portable Path Configuration

The system automatically handles paths based on [`config.yaml`](config.yaml):

```python
from src.config import PathConfig

# Get configuration instance
config = PathConfig()

# Access any configured path
bom_file = config.get_file_path('current', 'bom')  # Returns pathlib.Path
data_dir = config.get_data_path('current')         # Returns pathlib.Path
test_baseline = config.get_test_baseline()         # Returns pathlib.Path

# All paths work cross-platform automatically
print(bom_file)  # Windows: WindowsPath('C:/project/data/current/bom.csv')
                 # Linux:   PosixPath('/project/data/current/bom.csv')
```

## Data Requirements

### BOM Explosion Input Files
- `bom.csv` - Bill of materials relationships
- `items.csv` - Item master data with lead times
- `independent_demand.csv` - Customer demand/forecast

### Working Calendar Input Files
- `calendar_rules.csv` - Working time rules by weekday
- `calendar_exceptions.csv` - Holiday and exception schedules

All input files should be placed in `./data/current/` directory.

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_explode_bom.py -v
python -m pytest tests/test_working_calendar.py -v
```

## Algorithms

### 1. BOM Explosion
- **Purpose**: Explodes Bill of Materials to calculate component demand
- **Features**: 
  - Multi-level BOM explosion
  - Lead time offset calculations
  - Date-based demand scheduling
  - CSV and Excel integration

### 2. Working Calendar
- **Purpose**: Calculates working time completion considering calendar rules
- **Features**:
  - Custom calendar definitions
  - Holiday and exception handling
  - Excel data type compatibility
  - Optimized interval-based calculations

## Excel Integration

Both modules support Excel integration via xlwings:
- **BOM Explosion**: `bom_explosion_from_excel()` function
- **Working Calendar**: `@func` decorated Excel functions

## Dependencies

- **`pandas`** - Data manipulation and CSV handling
- **`pytest`** - Testing framework
- **`pyyaml`** - YAML configuration parsing for portable paths
- **`xlwings`** - Excel integration (optional)

## Cross-Platform Portability Features

This repository is designed to be fully portable across operating systems:

### Path Management
- **YAML Configuration**: Single source of truth for all file paths in [`config.yaml`](config.yaml)
- **pathlib.Path**: Cross-platform path handling (Windows backslashes, Unix forward slashes)
- **Automatic Root Detection**: Finds project root regardless of execution directory
- **Relative Path Resolution**: All paths resolved relative to project root

### Dataset Flexibility
- **Multiple Datasets**: Support for `current`, `test`, and custom dataset configurations
- **Configurable Structure**: Easy to add new datasets or modify existing ones
- **Consistent API**: Same function calls work across all datasets

### Import Compatibility
- **Package vs Script**: Modules work both as packages (from tests) and direct scripts
- **Fallback Imports**: Automatic fallback between relative and absolute imports
- **Cross-Platform Execution**: Identical behavior on Windows, macOS, and Linux

## Development

This repository is designed to be:
- **Portable**: Cross-platform compatible with YAML-based path configuration
- **Simple**: Two core modules with centralized configuration management
- **Well-tested**: Comprehensive test coverage with portable test data
- **Documented**: Clear usage examples and cross-platform API documentation
- **Maintainable**: DRY principle with single source of truth for paths

## License

This project is designed for educational and development purposes.