# PyPS Scheduling Algorithms

Two production-ready planning algorithms with Excel integration: **BOM Explosion** and **Working Calendar** calculations.

## 🚀 Quick Start

### Option 1: Git Clone (Recommended)
**PowerShell/Command Prompt (Windows):**
```powershell
git clone https://github.com/your-repo/pyps_algorithms.git
cd pyps_algorithms
pip install pandas pytest xlwings pyyaml
python main.py
```

**Bash (Linux/macOS):**
```bash
git clone https://github.com/your-repo/pyps_algorithms.git
cd pyps_algorithms
pip install pandas pytest xlwings pyyaml
python main.py
```

### Option 2: Download ZIP
1. [Download ZIP](https://github.com/your-repo/pyps_algorithms/archive/main.zip)
2. Extract to your desired location
3. Open PowerShell/Command Prompt in the extracted folder
4. Run:
   ```powershell
   pip install pandas pytest xlwings pyyaml
   python main.py
   ```

## 📋 What You Get

### 1. **BOM Explosion Algorithm**
Explodes Bill of Materials with lead time calculations for production planning.

### 2. **Working Calendar Algorithm** 
Calculates working time completion dates considering business calendars, holidays, and exceptions.

### 3. **Excel Integration Ready**
Both algorithms include **xlwings lite** modules for seamless Excel integration.

## 🏗️ Project Structure

```
pyps_algorithms/
├── src/                        # Main algorithms
│   ├── config.py               # Configuration management
│   ├── explode_bom.py          # BOM explosion algorithm
│   └── working_calendar.py     # Working calendar algorithm
├── xlwings_lite/               # Excel-compatible versions
│   ├── explode_bom_lite.py     # BOM explosion for Excel
│   └── working_calendar_lite.py # Working calendar for Excel
├── excel/                      # Excel workbook examples
│   └── Archive/                # Previous Excel versions
├── data/current/               # Sample input data
│   ├── bom.csv                 # Bill of materials
│   ├── items.csv               # Item master data
│   ├── independent_demand.csv  # Customer demand
│   ├── calendar_rules.csv      # Working time rules
│   └── calendar_exceptions.csv # Holidays & exceptions
├── tests/                      # Comprehensive test suite
├── config.yaml                 # Path configuration
└── main.py                     # Interactive demo
```

## 🎯 Getting Started Examples

### Run the Interactive Demo
**PowerShell/Command Prompt:**
```powershell
python main.py
```

**Bash:**
```bash
python main.py
```

### Try Individual Algorithms
**PowerShell/Command Prompt:**
```powershell
# BOM Explosion
python src/explode_bom.py

# Working Calendar
python src/working_calendar.py
```

**Bash:**
```bash
# BOM Explosion
python src/explode_bom.py

# Working Calendar
python src/working_calendar.py
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

```powershell
# Run consistency tests
python -m pytest tests/test_ast_consistency.py -v
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

## 🧪 Testing

**PowerShell/Command Prompt:**
```powershell
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_explode_bom.py -v
python -m pytest tests/test_working_calendar.py -v

# Run AST consistency tests
python -m pytest tests/test_ast_consistency.py -v
```

**Bash:**
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_explode_bom.py -v
python -m pytest tests/test_working_calendar.py -v

# Run AST consistency tests
python -m pytest tests/test_ast_consistency.py -v
```

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

This project is designed for educational and development purposes.