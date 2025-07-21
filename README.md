# PyPS Planning Algorithms

An repository with example planning and scheduling algorithms that can work as a component or within an Excel workbook.

1. BOM Explosion - Explodes Bill of Materials with lead time calculations
2. Working Calendar - Calculates working time completion with calendar rules and exceptions
3. Forecast ETS - Exponential Smoothing forecasting with comprehensive validation and CSV output

The core functions are included in ./src along with companion modules designed to function in xlwings lite (except Forecast ETS which is designed for local execution with CSV integration).

## 🚀 Quick Start

### Option 1: Git Clone (If Git installed)
**Cross-platform:** All commands below work in bash, zsh, and PowerShell
```bash
git clone https://github.com/kinetica-partners/pyps-algorithms
cd pyps_algorithms
uv sync
uv run python src/explode_bom.py
uv run python src/working_calendar.py
```

### Option 2: Download ZIP
1. **Download** the repository zip file (https://github.com/kinetica-partners/pyps-algorithms/archive/refs/heads/main.zip)
2. **Extract** to your desired location
3. **Rename** the folder from `pyps-algorithms-main` to `pyps_algorithms`
4. **Run** (works in bash, zsh, and PowerShell):
```bash
uv sync
uv run python src/explode_bom.py
uv run python src/working_calendar.py
```

## 📋 What You Get

### 1. **BOM Explosion Algorithm**
Explodes Bill of Materials with lead time calculations for production planning.

### 2. **Working Calendar Algorithm**
Calculates working time completion dates considering business calendars, holidays, and exceptions.

### 3. **Forecast ETS Algorithm**
Exponential Smoothing (ETS) time series forecasting with automatic data generation, comprehensive validation, and CSV output for Excel integration.

### 4. **Excel Integration Ready**
BOM Explosion and Working Calendar algorithms include **xlwings lite** modules for seamless Excel integration. Forecast ETS is designed for local execution with CSV output that integrates with `excel/Forecast_ETS_v01.01.xlsm`.

## 🏗️ Project Structure

```
pyps_algorithms/
├── src/                              # Main algorithms
│   ├── config.py                     # Configuration management
│   ├── explode_bom.py                # BOM explosion algorithm
│   ├── working_calendar.py           # Working calendar algorithm
│   └── forecast_ets.py               # ETS forecasting algorithm
├── xlwings_lite/                     # Excel-compatible versions
│   ├── explode_bom_lite.py           # BOM explosion for Excel
│   └── working_calendar_lite.py      # Working calendar for Excel
├── excel/                            # Excel workbook examples
│   ├── BOM_Explosion_v01.xx.xlsm     # BOM explosion workbook
│   ├── Working_Calendar_v01.xx.xlsm  # Working calendar workbook
│   └── Forecast_ETS_v01.xx.xlsm      # ETS forecasting integration
├── data/current/                     # Sample input data
│   ├── bom.csv                       # Bill of materials
│   ├── items.csv                     # Item master data
│   ├── independent_demand.csv        # Customer demand
│   ├── calendar_rules.csv            # Working time rules
│   └── calendar_exceptions.csv       # Holidays & exceptions
├── tests/                            # Comprehensive test suite
├── config.yaml                       # Path configuration
```

## 🎯 Getting Started Examples

### Try Individual Algorithms
**Cross-platform:** All commands below work in bash, zsh, and PowerShell
```bash
# BOM Explosion
uv run python src/explode_bom.py

# Working Calendar
uv run python src/working_calendar.py

# ETS Forecasting (with comprehensive test data generation)
uv run python -m pytest tests/test_forecast_ets.py -v
```

## 📊 Excel Integration with xlwings

### Step 1: Install xlwings Add-in
1. **Install from Microsoft AppSource**: [xlwings Add-in](https://appsource.microsoft.com/en-us/product/office/WA200001351)
2. **Alternative**: Install via xlwings CLI:
   ```powershell
   xlwings addin install
   ```

### Step 2: Using the Pre-built Excel Examples
The `excel/` folder contains ready-to-use Excel workbooks with the algorithms already integrated. These examples demonstrate both algorithms working with real Excel data.

### Step 3: Creating Your Own Excel Integration

#### Important: One Module Per Workbook
⚠️ **xlwings Limitation**: Each Excel workbook can only contain **ONE** xlwings lite module. Do not attempt to paste multiple `_lite.py` modules into the same Excel file.

#### Using xlwings Lite Modules
The `xlwings_lite/` folder contains self-contained modules you can add to new Excel workbooks:

**For BOM Explosion:**
1. Create a new Excel workbook
2. Copy the entire contents of `xlwings_lite/explode_bom_lite.py`
3. In Excel, press `Alt + F11` to open VBA editor
4. Insert a new module and paste the Python code
5. Use the `@func` decorated functions directly in Excel cells

**For Working Calendar:**
1. Create a new Excel workbook  
2. Copy the entire contents of `xlwings_lite/working_calendar_lite.py`
3. Follow the same VBA integration steps
4. Use functions like `=calculate_working_completion_time()` in Excel

## 🔄 Dual Module Architecture

### Why Two Versions of Each Algorithm?

This project maintains **dual modules** for each algorithm:

1. **Main modules** (`src/`): Full-featured with file I/O, configuration, and testing
2. **Lite modules** (`xlwings_lite/`): Self-contained, Excel-compatible versions

### Key Differences:

| Feature | Main Modules | Lite Modules |
|---------|-------------|--------------|
| **Dependencies** | Uses `config.yaml` and file system | Self-contained, no external files |
| **Data Input** | Reads from CSV files | Accepts Excel data ranges |
| **Error Handling** | Comprehensive with file validation | Simplified for Excel compatibility |
| **Testing** | Full test suite coverage | Validated through AST consistency tests |
| **Excel Integration** | Requires file paths | Direct Excel data integration |

### AST Consistency Testing

The project uses **AST (Abstract Syntax Tree) consistency tests** to ensure both versions remain functionally identical:

```bash
# Run consistency tests
uv run python -m pytest tests/test_ast_consistency.py -v
```

This ensures that while the modules are **not DRY** (Don't Repeat Yourself), they are **algorithmically identical** and maintained through automated testing.

## 📝 Algorithm Usage Examples

### BOM Explosion
```python
# Main module (file-based)
from src.explode_bom import bom_explosion_from_csv
result = bom_explosion_from_csv(dataset='current')

# Lite module (Excel-compatible)
from xlwings_lite.explode_bom_lite import explode_bom_iterative
result = explode_bom_iterative(bom_data, items_data, demand_data)
```

### Working Calendar
```python
# Main module (file-based)
from src.working_calendar import calculate_working_completion_time
completion = calculate_working_completion_time(
    start_datetime=datetime(2025, 1, 6, 9, 0),
    jobtime=120,  # minutes
    calendar_id="default"
)

# Lite module (Excel-compatible)
from xlwings_lite.working_calendar_lite import calculate_working_completion_time
completion = calculate_working_completion_time(
    start_datetime=44927.375,  # Excel date serial
    jobtime=120,
    calendar_rules_data=rules_range,
    calendar_exceptions_data=exceptions_range
)
```

### ETS Forecasting
```python
# Main module (generates comprehensive test data and forecasts)
from src.forecast_ets import generate_forecast_ets_weekly

# Generate forecasts for time series data
forecast_df = generate_forecast_ets_weekly(
    series=time_series_data,  # DataFrame with 'item', 'period', 'quantity' columns
    forecast_range=26,        # 26 weeks ahead
    trend="add",             # Additive trend
    seasonal="add",          # Additive seasonality
    seasonal_periods=52      # 52 weeks seasonal cycle
)

# Note: No xlwings lite version - designed for local execution with CSV output
# Integrates with excel/Forecast_ETS_v01.01.xlsm via CSV files
```

## 🧪 Testing

**Cross-platform:** All commands below work in bash, zsh, and PowerShell
```bash
# Run all tests
uv run python -m pytest tests/ -v

# Run specific test modules
uv run python -m pytest tests/test_explode_bom.py -v
uv run python -m pytest tests/test_working_calendar.py -v
uv run python -m pytest tests/test_forecast_ets.py -v

# Run AST consistency tests
uv run python -m pytest tests/test_ast_consistency.py -v
```

### ETS Forecasting Tests & Data Generation

The ETS forecasting module includes **automatic test data generation** with comprehensive validation:

```bash
# Run ETS tests (automatically generates training, test, and forecast CSV files)
uv run python -m pytest tests/test_forecast_ets.py -v
```

This automatically creates:
- `tests/test_data/ets_training_data.csv` - 936 training records (6 items × 156 weeks)
- `tests/test_data/ets_test_data.csv` - 156 test records for validation
- `tests/test_data/ets_forecast_data.csv` - 156 forecast records for Excel integration

**Time Series Patterns Generated:**
- **Simple Patterns**: Sinusoidal seasonality with linear trends
- **Complex Patterns**: Dual-peak seasonality with non-linear trends and higher noise
- **Validation**: %MAE criteria from 15-50% based on pattern complexity

## 📋 Data Requirements

### BOM Explosion Input Files
- `bom.csv` - Bill of materials relationships
- `items.csv` - Item master data with lead times
- `independent_demand.csv` - Customer demand/forecast

### Working Calendar Input Files
- `calendar_rules.csv` - Working time rules by weekday
- `calendar_exceptions.csv` - Holiday and exception schedules

All input files are provided in `./data/current/` for testing.

## 🔧 Dependencies

- **`pandas`** - Data manipulation and CSV handling
- **`pytest`** - Testing framework
- **`pyyaml`** - YAML configuration parsing
- **`xlwings`** - Excel integration
- **`statsmodels`** - ETS forecasting models
- **`numpy`** - Numerical computations for forecasting

## 🏆 Key Features

### Production-Ready Algorithms
- **BOM Explosion**: Multi-level explosion with lead time calculations
- **Working Calendar**: Business calendar with holidays and exceptions

### Excel Integration
- **Self-contained modules** for easy Excel integration
- **Compatible with xlwings lite** for seamless Excel functions
- **Pre-built Excel examples** ready to use

### Dual Architecture Benefits
- **Main modules**: Full testing and file integration
- **Lite modules**: Excel-compatible with minimal dependencies
- **AST consistency**: Automated testing ensures identical functionality

### Cross-Platform Support
- **Windows PowerShell** and **Command Prompt** compatible
- **Linux/macOS Bash** compatible
- **Automatic path resolution** for any operating system

## 📄 License

## License

This project is licensed under the MIT License.

### Third-Party Dependencies

- **xlwings**: BSD 3-clause license (compatible with MIT)
- **pandas**: BSD 3-clause license (compatible with MIT)
- **pytest**: MIT license (compatible with MIT)
