# xlwings Lite : Coding Guidelines for AI Coders (Humans Welcome)
### Last Update date: 25th June 2025

## Table of Contents
1. [Introduction](#1-introduction)
2. [xlwings Lite Interface Overview](#2-xlwings-lite-interface-overview)
3. [AI Coder Checklist: Core Directives for All Scripts](#3-ai-coder-checklist-core-directives-for-all-scripts)
4. [Compatibility](#4-compatibility)
   - [Supported Platforms](#41-supported-platforms)
     - [Windows Desktop](#windows-desktop)
     - [macOS Desktop](#macos-desktop)
     - [Excel on the Web](#excel-on-the-web)
5. [Editor Features](#5-editor-features)
   - [Core Features](#51-core-features)
     - [AutoSave Functionality](#511-autosave-functionality)
     - [Keyboard Shortcuts](#512-keyboard-shortcuts)
     - [Code Completion](#513-code-completion)
     - [Output Pane](#514-output-pane)
     - [Standalone Mode](#515-standalone-mode)
6. [Custom Functions](#6-custom-functions)
    - [Basic Syntax](#61-basic-syntax)
    - [Working with DataFrames](#62-working-with-dataframes)
    - [Type Hints Support](#63-type-hints-support)
    - [Variable Arguments](#64-variable-arguments)
    - [Documentation](#65-documentation)
    - [Date/Time Handling](#66-datetime-handling)
7. [Custom Scripts](#7-custom-scripts)
    - [Basic Syntax](#71-basic-syntax)
    - [Running Scripts](#72-running-scripts)
    - [Sheet Buttons](#73-sheet-buttons)
    - [Configuration Options](#74-configuration-options)
    - [Tips and Troubleshooting](#75-tips-and-troubleshooting)
8. [Comprehensive Guide to Limitations & Unsupported Features](#8-comprehensive-guide-to-limitations-unsupported-features)
   - [Pyodide and Environment Constraints](#81-pyodide-and-environment-constraints)
   - [Unsupported xlwings API Features](#82-unsupported-xlwings-api-features)
   - [Planned Future Enhancements](#83-planned-future-enhancements)
9. [Connecting to External Data & APIs](#9-connecting-to-external-data-apis)
   - [Working with Web APIs](#91-working-with-web-apis)
   - [Connecting to Databases via an API Layer](#92-connecting-to-databases-via-an-api-layer)
10. [Security Best Practices](#10-security-best-practices)
   - [Environment Variables for Secrets](#101-environment-variables-for-secrets)
   - [Cross-Origin Resource Sharing (CORS)](#102-cross-origin-resource-sharing-cors)
11. [Python Dependencies Management](#11-python-dependencies-management)
12. [Latest Features](#12-latest-features-as-of-june-2025)
13. [Example Scripts](#13-example-scripts)
    - [Starter Examples from xlwings documentation](#1-starter-examples-from-xlwings-documentation)
    - [XGBoost Response Model (DEF_4K.xlsx)](#2-xgboost-response-model-def_4kxlsx)
    - [Credit Card Segment Analysis (RBICC_DEC2024.xlsx)](#3-credit-card-segment-analysis-rbicc_dec2024xlsx)
    - [Database Integration](#4-database-integration)
    - [Web Scraping with LLM Processing (URL_LIST.xlsx)](#5-web-scraping-with-llm-processing-url_listxlsx)
    - [Advanced EDA and Schema Analysis with LLMs](#6-advanced-eda-and-schema-analysis-with-llms)


## 1. Introduction
This repository contains example scripts demonstrating how to use xlwings Lite for data analysis and machine learning directly within Excel. The documentation provides comprehensive guidance for both basic and advanced usage scenarios.

## 2. xlwings Lite Interface Overview
**CRITICAL FOR AI CODERS:** xlwings Lite (released January 2025) integrates as a task pane within Excel. Understanding this interface is essential for effective user guidance.

### 2.1 Interface Components
- **Task Pane Location:** Right side of Excel window, appears as native Excel feature
- **Header:** xlwings Lite logo with minimize/close controls
- **Script Execution Area:** 
  - Green play button with script name (e.g., "add_derived_metrics")
  - Dropdown arrow for script selection
  - One-click execution via F5 or button click
- **Code Editor Tabs:**
  - `main.py` - Primary Python script file
  - `requirements.txt` - Package dependencies
  - Line numbers and syntax highlighting included
- **Console Log Area:** Below code editor, displays print() output and error messages

### 2.2 Key Integration Points
- **Seamless Excel Integration:** Task pane maintains Excel's native look/feel
- **Live Code Editing:** Direct Python code editing within Excel environment  
- **Context Awareness:** Scripts can access and manipulate active worksheet data
- **Real-time Feedback:** Console immediately shows script output and errors
- **Multi-file Support:** Both Python code and dependency management in single interface

### 2.3 User Guidance Framework
When directing users:
- **Code Location:** "In the xlwings Lite task pane on the right, click the main.py tab"
- **Package Issues:** "Add missing packages to requirements.txt tab in xlwings Lite pane"
- **Error Diagnosis:** "Check the console log area below the code editor for error details"
- **Script Execution:** "Click the green play button or press F5 to run the script"

## 3. AI Coder Checklist: Core Directives for All Scripts

{: .golden-rules}
**Golden Rules: These 5 directives are non-negotiable and MUST be applied in every script.**

1. **ALWAYS** use the `find_table_in_workbook()` helper to locate tables.
2. **ALWAYS** use `.options(index=False)` when writing DataFrames.
3. **CRITICAL: NEVER** use `.expand()` on newly written data. It runs too fast and **WILL** fail with an `IndexError`. **ALWAYS** define ranges explicitly using `.resize(df.shape[0] + 1, df.shape[1])`). This is the most common point of failure.
4. **ALWAYS** wrap fallible operations like `sheet.tables.add()` in `try...except`.
5. **ALWAYS** use hex strings (e.g., '#F0F0F0') for colors, **NEVER** RGB tuples which will raise `ValueError`.

### 2.1 Script Robustness & Reliability
> - **Anticipate Failures:** **Always** wrap potentially fallible operations like creating Excel tables (`sheet.tables.add(...)`) or formatting calls in `try...except` blocks to prevent minor issues from halting the entire script.
>   - **Incorrect (Brittle):** `sheet.tables.add(range_to_format)`
>   - **Correct (Robust):**
>     ```python
>     try:
>         sheet.tables.add(range_to_format)
>     except Exception as e:
>         print(f"⚠️ Could not create table: {e}")
>     ```
>
> - **Handle DataFrame Index:** When writing a pandas DataFrame to Excel, **always** use `.options(index=False)` to prevent writing the DataFrame's index as an unwanted column. This prevents confusion and maintains clean data presentation.
>   - **Incorrect (Adds Index):** `sheet["A1"].value = df`
>   - **Correct (No Index):** `sheet["A1"].options(index=False).value = df`
>
> - **Manage Existing Sheets:** Before creating a new sheet, always check if a sheet with the same name already exists and delete it to ensure the script is re-runnable.
>   - **Incorrect (Can Fail):** `new_sheet = book.sheets.add("Results")`
>   - **Correct (Safe):**
>     ```python
>     if "Results" in [s.name for s in book.sheets]:
>         book.sheets["Results"].delete()
>     new_sheet = book.sheets.add("Results")
>     ```
>
> - **Robustly Locate Excel Tables:** The `book.tables` attribute is not available in xlwings Lite, and relying on `book.sheets.active` makes scripts brittle and unreliable. To ensure scripts work regardless of which sheet is currently active, you **MUST** include and use the following helper function to find tables by name.
>
>   ```python
>   def find_table_in_workbook(book: xw.Book, table_name: str):
>       """Searches all sheets for a table with the given name."""
>       for sheet in book.sheets:
>           if table_name in sheet.tables:
>               return sheet.tables[table_name]
>       return None
>   ```
>
>   **This pattern is mandatory for all scripts accessing a named Excel table.**
>
>   - **Incorrect (Brittle) Method:** `table = book.sheets.active.tables['MyTable']`
>   - **Correct (Robust) Method:** `table = find_table_in_workbook(book, 'MyTable')`

### 2.2 Formatting & Readability
> - **Ensure Visible Headers:** When setting a background color for a cell or range (`.color`), you **MUST** also explicitly set a contrasting font color (`.font.color`) in the same step. For a light background, use a dark font. **CRITICAL:** Only use hex color strings (e.g., '#F0F0F0'), as RGB tuples are not supported and will raise a `ValueError`.
>   - **Incorrect (Will Raise ValueError):** `header_range.color = (240, 240, 240)`
>   - **Incorrect (Unreadable):** `header_range.color = '#F0F0F0'  # Missing font color`
>   - **Correct (Always Readable):**
>     ```python
>     # ALWAYS use hex strings for both background and font colors.
>     # RGB tuples for .color are not supported and WILL cause a ValueError.
>     header_range.color = '#F0F0F0'         # Light gray background
>     header_range.font.color = '#000000'    # Black text
>     ```
>
> - **Use Clean Column Names:** Before writing a DataFrame to Excel, proactively rename columns for professional presentation (e.g., `df.rename(columns={'raw_name': 'Clean Name'})`).
>
> - **Narrate the Script's Progress:** Use descriptive `print()` statements at each major step of the script. This gives the user confidence and critical information if something goes wrong.
>
> - **Create Formal Excel Tables:** When writing a DataFrame (especially a summary) to a new region on a sheet, you **MUST** convert it into a formal Excel Table. Simply writing the data and coloring the header is insufficient and produces unprofessional results.
>   - **CRITICAL: Range Sizing:** Define the table's range explicitly using `.resize()` with the DataFrame's shape (`df.shape[0] + 1` for rows, `df.shape[1]` for columns). **NEVER** use `.expand()` on newly written data as it runs too fast and **WILL** fail with an `IndexError` before Excel can register the data.
>   - **Incorrect (WILL Fail):** `range_to_format = sheet["B2"].expand('down')`
>   - **Correct (Always Works):** `range_to_format = sheet["B2"].resize(df.shape[0] + 1, df.shape[1])`
>   - Wrap in `try...except`: As a fallible operation, the `sheet.tables.add()` call must be wrapped in a `try...except` block to ensure the script doesn't halt if table creation fails.
>   - For a best-practice implementation, see the table creation logic in the `XGBoost Response Model` script.

### 2.3 Known Limitations to Acknowledge
> - **Font Properties:** Font properties (`bold`, `italic`, `color`, `size`, `name`) can be **set**, but they cannot be **read**.
>
> - **Sheet Autofit:** `Sheet.autofit()` is not supported and will raise a `NotImplementedError`.
>
> - **Custom Script Arguments:** Custom scripts (decorated with `@script`) can only accept a single argument: `book: xw.Book`.
>
> - **No Direct API Access:** The `.api` property is not available.

## 4. Compatibility
### 4.1 Supported Platforms
1. **Windows Desktop**:
   - Microsoft 365
   - Office 2021 or later

2. **macOS Desktop**:
   - Microsoft 365
   - Office 2021 or later
   - Requires macOS Ventura (macOS 13) or later

3. **Excel on the Web**:
   - Works with any modern browser
   - Compatible with free version of Excel
   - Access via Microsoft 365 or [free Excel online](https://www.microsoft.com/en-us/microsoft-365/free-office-online-for-the-web)

## 5. Editor Features
### 5.1 Core Features
xlwings Lite uses a VS Code-based editor with many familiar features:

#### 5.1.1 AutoSave Functionality
- Changes automatically written to workbook
- Green checkmark indicates active tab status
- With OneDrive/SharePoint AutoSave: continuous saving
- Without AutoSave: saves on workbook save

#### 5.1.2 Keyboard Shortcuts
| Action | Windows/Linux | macOS |
|--------|--------------|-------|
| Move line up/down | `Alt + ↑/↓` | `Alt + ↑/↓` |
| Delete line | `Shift + Ctrl + K` | `Shift + ⌘ + K` |
| Multi-cursor above/below | `Alt + Ctrl + ↑/↓` | `Alt + ⌘ + ↑/↓` |
| Format with Black | `Shift + Alt + F` | `Shift + Alt + F` |
| Run script | `F5` | `F5` |
| Change font size | `Ctrl + +/-` | `⌘ + +/-` |

#### 5.1.3 Code Completion
- Basic code completion available
- Note: Currently limited for NumPy
- Complex packages (like pandas) may have initial delay

#### 5.1.4 Output Pane
- Resizable vertical pane
- Shows `print()` output
- Displays full error tracebacks

#### 5.1.5 Standalone Mode
- Editor can be dragged out of Excel
- Provides VBA-like separate window experience

## 6. Custom Functions
### 6.1 Basic Syntax
```python
from xlwings import func

@func
def hello(name):
    return f"Hello {name}!"
```
Call in Excel with: `=HELLO("World")` or `=HELLO(A1)`

### 6.2 Working with DataFrames
```python
import pandas as pd
from xlwings import func, arg, ret

@func
@arg("df", pd.DataFrame)
@ret(index=False, header=False)
def correl2(df):
    return df.corr()
```

### 6.3 Type Hints Support
```python
from xlwings import func
import pandas as pd

@func
def myfunction(df: pd.DataFrame) -> pd.DataFrame:
    return df
```

### 6.4 Variable Arguments
```python
from xlwings import func, arg

@func
@arg("*args", pd.DataFrame, index=False)
def concat(*args):
    return pd.concat(args)
```

### 6.5 Documentation
```python
from xlwings import func, arg

@func
@arg("name", doc='A name such as "World"')
def hello(name):
    """This is a classic Hello World example"""
    return f"Hello {name}!"
```

### 6.6 Date/Time Handling
```python
import datetime as dt
from xlwings import func

@func
@arg("date", dt.datetime)
def process_date(date):
    return date

# For multiple dates in a range
import xlwings as xw
@func
def process_dates(dates):
    return [xw.to_datetime(d) for d in dates]

# For DataFrames with dates
@func
@arg("df", pd.DataFrame, parse_dates=[0])
def timeseries_start(df):
    return df.index.min()
```

## 7. Custom Scripts
### 7.1 Basic Syntax
Custom Scripts in xlwings Lite are Python functions that run at the click of a button and have access to the Excel object model. They are equivalent to VBA Subs or Office Scripts.

```python
import xlwings as xw
from xlwings import script

@script
def hello_world(book: xw.Book):
    sheet = book.sheets[0]
    sheet["A1"].value = "Hello xlwings!"
```

### 7.2 Running Scripts
- Click the run button or press F5 in the xlwings Lite add-in
- Select different scripts from the dropdown menu
- Changes to script names automatically update in the dropdown

### 7.3 Sheet Buttons
- Create buttons using Excel shapes with hyperlinks
- Name the shape in the name box (e.g., `xlwings_button`)
- Link the shape to a cell behind it (e.g., `B4`)
- Configure the script with:
```python
@script(button="[xlwings_button]Sheet1!B4", show_taskpane=True)
def hello_world(book: xw.Book):
    # your code here
```
Note: Button clicks change cell selection, so don't use for scripts that depend on selected cells.

### 7.4 Configuration Options
```python
@script(
    include=["Sheet1", "Sheet2"],  # Only include these sheets' content
    exclude=["BigData"],  # Exclude these sheets' content
    button="[mybutton]Sheet1!A1",  # Sheet button configuration
    show_taskpane=True  # Show taskpane when button clicked
)
```

### 7.5 Tips and Troubleshooting
- Use `include`/`exclude` to limit data transfer for large workbooks
- Only include sheets needed by your script
- Don't select the linked cell initially
- Verify button name in script decorator matches exactly
- Restart xlwings Lite to re-register event handlers
- Excel web doesn't support adding shape hyperlinks (but works if set up in desktop)

## 8. Comprehensive Guide to Limitations & Unsupported Features
This section provides a consolidated overview of all known limitations in xlwings Lite as of June 2025. Understanding these constraints is crucial for effective development.

### 8.1 Pyodide and Environment Constraints
xlwings Lite runs on Pyodide, which imposes several environment-level restrictions:
- **Python Version**: The Python version is fixed by the specific Pyodide distribution used in the add-in.
- **Memory Limit**: There is a 2GB memory limit for the Python environment.
- **Debugging**: There is no debugger support. Use `print()` statements to the Output Pane for debugging.
- **Concurrency**: `multiprocessing` and `threading` are not supported.
- **Package Availability**: Only packages that are pure Python or have been specifically compiled for the Pyodide environment can be used. Check the [official Pyodide packages list](https://pyodide.org/en/stable/usage/packages-in-pyodide.html) for availability.
- **Network Connections**: Direct TCP/IP sockets are not available. This means:
    - No direct connections to databases like PostgreSQL, MySQL, etc. (must use a web API layer).
    - All HTTP requests are subject to browser CORS (Cross-Origin Resource Sharing) policies.

### 8.2 Unsupported xlwings API Features
Many features from the classic xlwings API are not yet implemented in xlwings Lite. The following is a non-exhaustive list of common, unsupported properties and methods:

```python
# App limitations
xlwings.App:
    - cut_copy_mode
    - quit()
    - display_alerts
    - startup_path
    - calculate()
    - status_bar
    - path
    - version
    - screen_updating
    - interactive
    - enable_events
    - calculation

# Book limitations
xlwings.Book:
    - to_pdf()
    - save()

# Characters limitations
xlwings.Characters:
    - font
    - text

# Chart limitations
xlwings.Chart:
    - set_source_data()
    - to_pdf()
    - parent
    - delete()
    - top, width, height, left
    - name
    - to_png()
    - chart_type

xlwings.Charts:
    - add()

# Font limitations (setting supported as of April 2025, getting isn't)
xlwings.Font:
    - size
    - italic
    - color
    - name
    - bold

# Note limitations
xlwings.Note:
    - delete()
    - text

# PageSetup limitations
xlwings.PageSetup:
    - print_area

# Picture limitations
xlwings.Picture:
    - top
    - left
    - lock_aspect_ratio

# Range limitations
xlwings.Range:
    - hyperlink
    - formula
    - font
    - width
    - formula2
    - characters
    - to_png()
    - columns
    - height
    - formula_array
    - paste()
    - rows
    - note
    - merge_cells
    - row_height
    - get_address()
    - merge()
    - to_pdf()
    - autofill()
    - top
    - wrap_text
    - merge_area
    - column_width
    - copy_picture()
    - table
    - unmerge()
    - current_region
    - left

# Shape limitations
xlwings.Shape:
    - parent
    - delete()
    - font
    - top
    - scale_height()
    - activate()
    - width
    - index
    - text
    - height
    - characters
    - name
    - type
    - scale_width()
    - left

# Sheet limitations
xlwings.Sheet:
    - page_setup
    - used_range
    - shapes
    - charts
    - autofit()
    - copy()
    - to_html()
    - select()
    - visible

# Table limitations
xlwings.Table:
    - display_name
    - show_table_style_last_column
    - show_table_style_column_stripes
    - insert_row_range
    - show_table_style_first_column
    - show_table_style_row_stripes
```

### 8.3 Planned Future Enhancements
The following features are on the development roadmap but are **not yet available** as of June 2025.

- **File System Access**:
  - ❌ No local file access
  - ❌ No direct file system operations
  - 🔄 Planned: Enable access to local files

- **Development Features**:
  - ❌ No interactive Python terminal
  - ❌ No multiple Python modules
  - ❌ No external code storage
  - ❌ Limited code completion
  - ❌ No dark mode
  - 🔄 Planned: All these features in development

- **Excel Integration**:
  - ❌ No streaming functions
  - ❌ No object handles
  - ❌ Can't use Excel calculated values in same script
  - ❌ Limited formatting and charting
  - 🔄 Planned: Improved Excel object model coverage

- **Advanced Features**:
  - ❌ No Git integration
  - ❌ No Jupyter/marimo notebook support
  - ❌ No backend server option
  - ❌ Fixed Pyodide version
  - 🔄 Planned: All these features in roadmap

> **Note:** When users request unavailable features, guide them to use available workarounds, consider alternative approaches, and watch for updates in newer versions.

## 9. Connecting to External Data & APIs
This section details how xlwings Lite interacts with external data sources, including web APIs and databases. Due to its browser-based environment (Pyodide), direct database connections are not supported; all interactions must occur via web APIs.

### 9.1 Working with Web APIs
xlwings Lite supports common Python HTTP libraries and Pyodide's native `pyfetch` for making web requests.

1.  **Supported Libraries**:
    *   `requests`: For synchronous HTTP requests.
    *   `httpx`, `aiohttp`: For asynchronous HTTP requests (requires `async/await` syntax).
    *   `pyfetch`: Pyodide's native asynchronous JavaScript fetch wrapper.

    ```python
    # Synchronous with requests
    import requests
    response = requests.get("https://api.example.com/data")
    
    # Async with aiohttp
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.example.com/data") as response:
            data = await response.json()
    ```

2.  **Handling API Responses**:
    *   For FastAPI servers returning file responses, use `await response.text()` to extract content.
    *   Pipe-delimited data (common in FastAPI file responses) can be parsed by splitting lines with `.split("
")` and columns with `.split("|")`.
    *   When working with RexDB server responses, process them as plain text rather than attempting to parse as JSON.

3.  **Best Practices for Web API Requests**:
    *   Always use HTTPS for API requests.
    *   Handle errors gracefully with `try...except` blocks.
    *   Log detailed error information for debugging.
    *   Consider implementing request retries for reliability.

### 9.2 Connecting to Databases via an API Layer
Direct SQL database connections are not supported in xlwings Lite due to browser security restrictions. All database interactions must be mediated through a web API layer.

1.  **Custom API Layer**:
    *   The most flexible approach is to build your own web API using a framework like FastAPI or Flask. This allows for full control over authentication, query logic, and data shaping.
    *   This approach is used for the comprehensive examples in this document. For a detailed, end-to-end implementation showing how to list tables, get metadata, and query data, please see the **[Database Integration](#database-integration)** scripts in the Examples section.

2.  **Ready-to-Use Database REST APIs**:
    *   **PostgreSQL**: [PostgREST](https://docs.postgrest.org/), [Supabase](https://supabase.com/)
        ```python
        @script
        async def db_supabase(book: xw.Book):
            key = "<SUPABASE_KEY>"
            url = "https://<PROJECT>.supabase.co/rest/v1/<QUERY>"
            headers = {
                "apikey": key,
                "Authorization": f"Bearer {key}",
            }
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            print(data)
                except Exception as e:
                    print(f"Unexpected error: {e}")
        ```
    *   **Oracle**: [Oracle REST Data Services (ORDS)](https://www.oracle.com/database/technologies/appdev/rest.html)
    *   **MySQL**: [MySQL REST Service (MRS)](https://dev.mysql.com/doc/dev/mysql-rest-service/latest)
    *   **Other Options**: [NocoDB](https://nocodb.com/), [InfluxDB](https://www.influxdata.com/), [CouchDB](https://couchdb.apache.org/)

3.  **SQLite for Local/Network Data**:
    *   SQLite databases can be downloaded from a network location and processed locally within the Pyodide environment.
    *   Add `sqlite3` to `requirements.txt`.
    *   This method is suitable for smaller, self-contained datasets.
    ```python
    @script
    def process_sqlite(book: xw.Book):
        import sqlite3
        conn = sqlite3.connect('path/to/database.db')
        df = pd.read_sql('SELECT * FROM my_table', conn)
        conn.close()
        return df
    ```


## 10. Security Best Practices
Security is paramount when working with xlwings Lite, especially given its browser-based execution environment. This section outlines best practices for managing sensitive information and securing your API interactions.

### 10.1 Environment Variables for Secrets
xlwings Lite runs in a secure browser sandbox and cannot directly access local system environment variables. It provides two ways to set environment variables:

-   **Add-in Scope (Recommended for Secrets)**:
    -   Stored in the browser's local storage.
    -   Available across all workbooks.
    -   Never leaves your machine.
    -   **Use for API keys and sensitive secrets.**
    -   *Note*: These are cleared when the Office cache is cleared, so make backups!

-   **Workbook Scope**:
    -   Stored directly within the current workbook.
    -   **Not recommended for secrets** as they are embedded in the file.
    -   Specific to each workbook.

**Setting Environment Variables**:
1.  In the xlwings Lite add-in, navigate to the Environment Variables settings.
2.  Provide the Name (e.g., `OPENAI_API_KEY`), Value (e.g., `your-key`), and select the desired Scope (Add-in or Workbook).
3.  Click Save.
4.  Restart xlwings Lite for changes to take effect.

**Using Environment Variables in Code**:
```python
import os
import xlwings as xw
from xlwings import func, script

@script
def sample_script(book: xw.Book):
    key = os.getenv("OPENAI_API_KEY")
    if key is not None:
        print(key)
    else:
        raise Exception("Store your OPENAI_API_KEY key under Environment Variables!")
```

**Important Notes**:
-   Add-in scope overrides Workbook scope variables if names conflict.
-   Always back up important add-in scope variables.
-   Restart xlwings Lite after setting new variables to ensure they are loaded.

### 10.2 Cross-Origin Resource Sharing (CORS)
CORS is a browser security feature that restricts web pages from making requests to a different domain than the one that served the web page. Since xlwings Lite runs in the browser, all its HTTP requests are subject to CORS policies.

-   **CORS Requirements**:
    -   The target API server **must** explicitly allow requests from `https://addin.xlwings.org` (or your custom domain if self-hosting) via `Access-Control-Allow-Origin` headers.
    -   If you control the server, configure CORS headers in your API responses.
    -   If you do not control the server, consider using a CORS proxy (self-hosting is recommended for security over third-party services).

-   **Understanding HTTP Request Security from xlwings**:
    -   HTTP requests from xlwings in Excel originate from the origin `https://addin.xlwings.org`.
    -   Request headers will include:
        -   `origin: 'https://addin.xlwings.org'`
        -   `referer: 'https://addin.xlwings.org/'`
        -   `user-agent: '...(Windows NT ...)... Microsoft Edge WebView2...'`
    -   Requests go directly from the user's Excel/browser to your API server, **not** through `xlwings.org` servers.
    -   When using IP whitelisting on your API server, you must whitelist the **client's IP**, not `xlwings.org` servers.
    -   For token-based authentication, include the `Authorization` header in your `pyfetch` call.

    ```python
    # Example FastAPI setup with security headers logging and CORS configuration
    from fastapi import FastAPI, Request
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.middleware.cors import CORSMiddleware
    import logging

    logger = logging.getLogger(__name__)
    app = FastAPI()

    class HeadersLoggingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            origin = request.headers.get("origin", "No Origin")
            user_agent = request.headers.get("user-agent", "No UA")
            logger.info(f"Request from Origin: {origin}, UA: {user_agent}")
            return await call_next(request)
            
    app.add_middleware(HeadersLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://addin.xlwings.org"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    ```

## 11. Python Dependencies Management
xlwings Lite handles package management through the `requirements.txt` tab in the editor. Key points for AI coders to guide users:

1. **Installing Packages**:
   ```
   # Guide users to:
   1. Open xlwings Lite add-in
   2. Click on 'requirements.txt' tab
   3. Add package names and versions
   4. Installation starts automatically
   5. Check Output window for logs
   ```

2. **Package Compatibility**:
   - Packages must be Pyodide-compatible
   - Two sources checked:
     1. PyPI for pure Python wheels
     2. Pyodide's own repository for compiled packages
   - See [Pyodide packages list](https://pyodide.org/en/stable/usage/packages-in-pyodide.html)

3. **Version Pinning Rules**:
   ```python
   # Pure Python packages (including xlwings)
   xlwings==0.33.14
   requests==2.31.0

   # Pyodide-provided packages (don't pin!)
   pandas
   numpy
   ```

4. **Private Packages**:
   ```python
   # Can use direct URLs to wheels
   https://myserver.com/mypackage-1.0.0-py3-none-any.whl
   ```

5. **Important Notes**:
   - Restart xlwings Lite after changing package versions
   - Some popular packages (like PyTorch) not available
   - Custom builds possible but complex
   - Clear installation logs shown in Output window



## 12. Latest Features (as of June 2025)
Recent updates have added several important capabilities:

1. **Self-Hosting Support** (June 2025):
   - Build custom Docker images
   - Include additional packages
   - Self-host the add-in

2. **Sheet Button Support** (May 2025):
   - Create clickable buttons on sheets
   - Configure with `button` parameter
   - Requires xlwings 0.33.14+

3. **Performance Optimizations** (May 2025):
   - `include`/`exclude` configuration for scripts
   - Control workbook data transfer
   - Optimize for large workbooks

4. **Font Formatting** (April 2025):
   - Can now set font properties:
     - bold, italic, color
     - size, name
   - Note: Cannot read font properties

5. **Polars Support** (April 2025):
   - Native converter for Polars DataFrame
   - Native converter for Polars Series

6. **Bug Fixes and Improvements**:
   - Better error tracebacks in output pane
   - Fixed `Range.expand()`
  
## 13. Example Scripts

### 1. Starter Examples from xlwings documentation
Basic examples demonstrating xlwings Lite functionality:
- Hello World
- Seaborn visualization
- Custom function insertion
- Statistical operations

```python
@script
def hello_world(book: xw.Book):
    # Scripts require the @script decorator and the type-hinted
    # book argument (book: xw.Book)
    selected_range = book.selection
    selected_range.value = "Hello World!"
    selected_range.color = "#FFFF00"  # yellow


@script
def seaborn_sample(book: xw.Book):
    # Create a pandas DataFrame from a CSV on GitHub and print its info
    df = pd.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
    )
    print(df.info())

    # Add a new sheet, write the DataFrame out, and format it as Table
    sheet = book.sheets.add()
    sheet["A1"].value = "The Penguin Dataset"
    sheet["A3"].options(index=False).value = df
    sheet.tables.add(sheet["A3"].resize(len(df) + 1, len(df.columns)))

    # Add a Seaborn plot as picture
    plot = sns.jointplot(
        data=df, x="flipper_length_mm", y="bill_length_mm", hue="species"
    )
    sheet.pictures.add(plot.fig, anchor=sheet["B10"])

    # Activate the new sheet
    sheet.activate()


@script
def insert_custom_functions(book: xw.Book):
    # This script inserts the custom functions below
    # so you can try them out easily
    sheet = book.sheets.add()
    sheet["A1"].value = "This sheet shows the usage of custom functions"
    sheet["A3"].value = '=HELLO("xlwings")'
    sheet["A5"].value = "=STANDARD_NORMAL(3, 4)"
    sheet["A10"].value = "=CORREL2(A5#)"
    sheet.activate()
```



### 2. XGBoost Response Model (DEF_4K.xlsx)
This script demonstrates:
- Loading data from Excel tables
- Feature preparation and encoding
- XGBoost model training
- Model evaluation with ROC curves and decile tables
- Visualization and results export back to Excel

```python
@script
def score_and_deciles(book: xw.Book):
    print("📌 Step 1: Loading table 'DEF'...")

    sht = book.sheets.active
    table = sht.tables['DEF']
    df_orig = table.range.options(pd.DataFrame, index=False).value
    df = df_orig.copy()
    print(f"✅ Loaded table into DataFrame with shape: {df.shape}")

    # Step 2: Prepare features and target
    X = df.drop(columns=["CUSTID", "RESPONSE_TAG"])
    y = df["RESPONSE_TAG"].astype(int)
    print("🎯 Extracted features and target")

    # Step 3: One-hot encode
    X_encoded = pd.get_dummies(X, drop_first=True)
    print(f"🔢 Encoded features. Shape: {X_encoded.shape}")

    # Step 4: Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42
    )
    print(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Step 5: Train XGBoost
    model = XGBClassifier(max_depth=1, n_estimators=10, use_label_encoder=False,
                          eval_metric='logloss', verbosity=0)
    model.fit(X_train, y_train)
    print("🌲 Model trained successfully.")

    # Step 6: Score train/test
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]

    # Step 7: Gini
    train_gini = 2 * roc_auc_score(y_train, train_probs) - 1
    test_gini = 2 * roc_auc_score(y_test, test_probs) - 1
    print(f"📈 Train Gini: {train_gini:.4f}")
    print(f"📊 Test Gini: {test_gini:.4f}")

    # Step 8: Decile function
    def make_decile_table(probs, actuals):
        df_temp = pd.DataFrame({"prob": probs, "actual": actuals})
        df_temp["decile"] = pd.qcut(df_temp["prob"].rank(method="first", ascending=False), 10, labels=False) + 1
        grouped = df_temp.groupby("decile").agg(
            Obs=("actual", "count"),
            Min_Score=("prob", "min"),
            Max_Score=("prob", "max"),
            Avg_Score=("prob", "mean"),
            Responders=("actual", "sum")
        ).reset_index()
        grouped["Response_Rate(%)"] = round((grouped["Responders"] / grouped["Obs"]) * 100, 2)
        grouped["Cumulative_Responders"] = grouped["Responders"].cumsum()
        grouped["Cumulative_Response_%"] = round((grouped["Cumulative_Responders"] / grouped["Responders"].sum()) * 100, 2)
        return grouped

    train_decile = make_decile_table(train_probs, y_train)
    test_decile = make_decile_table(test_probs, y_test)
    print("📋 Created decile tables.")

    # Step 9: Insert deciles into new sheet
    print("📄 Preparing to insert decile tables into new sheet...")
    sheet_name = "DEF_Score_Deciles"
    existing_sheets = [s.name for s in book.sheets]

    if sheet_name in existing_sheets:
        try:
            book.sheets[sheet_name].delete()
            print(f"🧹 Existing '{sheet_name}' sheet deleted.")
        except Exception as e:
            print(f"⚠️ Could not delete existing sheet '{sheet_name}': {e}")

    new_sht = book.sheets.add(name=sheet_name, after=sht)
    new_sht["A1"].value = "Train Deciles"
    new_sht["A2"].value = train_decile

    start_row = train_decile.shape[0] + 4
    new_sht[f"A{start_row}"].value = "Test Deciles"
    new_sht[f"A{start_row+1}"].value = test_decile
    print(f"🗘️ Decile tables inserted into sheet '{sheet_name}'")

    # Step 10: Score full dataset and append as new column
    full_probs = model.predict_proba(X_encoded)[:, 1]
    df_orig["SCORE_PROBABILITY"] = full_probs
    table.range.options(index=False).value = df_orig
    print("✅ Appended SCORE_PROBABILITY to original table without changing its structure.")

    # Step 11: Create and insert graphs into Excel
    graph_sheet_name = "DEF_Score_Graphs"
    if graph_sheet_name in existing_sheets:
        try:
            book.sheets[graph_sheet_name].delete()
            print(f"🧹 Existing '{graph_sheet_name}' sheet deleted.")
        except Exception as e:
            print(f"⚠️ Could not delete existing sheet '{graph_sheet_name}': {e}")

    graph_sht = book.sheets.add(name=graph_sheet_name, after=new_sht)

    def plot_and_insert(fig, sheet, top_left_cell, name):
        try:
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"{name}.png")
            fig.savefig(temp_path, dpi=150)
            print(f"🖼️ Saved plot '{name}' to {temp_path}")
            anchor_cell = sheet[top_left_cell]
            sheet.pictures.add(temp_path, name=name, update=True, anchor=anchor_cell, format="png")
            print(f"✅ Inserted plot '{name}' at {top_left_cell}")
        except Exception as e:
            print(f"❌ Failed to insert plot '{name}': {e}")
        finally:
            plt.close(fig)

    # ROC Curve
    def plot_roc(y_true, y_prob, label):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

    fig1 = plt.figure(figsize=(6, 4))
    plot_roc(y_train, train_probs, "Train")
    plot_roc(y_test, test_probs, "Test")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plot_and_insert(fig1, graph_sht, "A1", name="ROC_Curve")

    # Cumulative Gain Curve
    def cumulative_gain_curve(y_true, y_prob):
        df = pd.DataFrame({'actual': y_true, 'prob': y_prob})
        df = df.sort_values('prob', ascending=False).reset_index(drop=True)
        df['cumulative_responders'] = df['actual'].cumsum()
        df['cumulative_pct_responders'] = df['cumulative_responders'] / df['actual'].sum()
        df['cumulative_pct_customers'] = (df.index + 1) / len(df)
        return df['cumulative_pct_customers'], df['cumulative_pct_responders']

    train_x, train_y = cumulative_gain_curve(y_train, train_probs)
    test_x, test_y = cumulative_gain_curve(y_test, test_probs)

    fig2 = plt.figure(figsize=(6, 4))
    plt.plot(train_x, train_y, label="Train")
    plt.plot(test_x, test_y, label="Test")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.title("Cumulative Gain (Decile) Curve")
    plt.xlabel("Cumulative % of Customers")
    plt.ylabel("Cumulative % of Responders")
    plt.legend()
    plt.grid(True)
    plot_and_insert(fig2, graph_sht, "A20", name="Gain_Curve")

    print(f"📊 Graphs added to sheet '{graph_sheet_name}'.")

    for sht in book.sheets:
        print(f"📄 Sheet found: {sht.name}")

    try:
        book.save()
        print("📅 Workbook saved successfully.")
    except Exception as e:
        print(f"❌ Failed to save workbook: {e}")
```

### 3. Credit Card Segment Analysis (RBICC_DEC2024.xlsx)
This script shows:
- Table formatting and data preparation
- Weighted scoring implementation
- Multicollinearity analysis
- Results visualization

```python
@script
def format_rbicc_table(book: xw.Book):
    print("Starting format_rbicc_table...")

    try:
        sht = book.sheets.active
        table = sht.tables['RBICC']
        rng = table.range
        print(f"Formatting table 'RBICC' on sheet '{sht.name}' at range: {rng.address}")

        # Get headers and all data
        headers = rng[0, :].value
        data_range = rng[1:, :rng.columns.count]
        print(f"Headers found: {headers}")

        # Define formatting rules by column name
        currency_cols = [col for col in headers if 'VALUE_AMT' in col or 'TICKET' in col]
        percent_cols = [col for col in headers if '_SHARE_' in col or 'RATIO' in col]
        round_cols = [col for col in headers if col in currency_cols + percent_cols]

        # Apply formatting column by column
        for col_idx, col_name in enumerate(headers):
            col_range = rng[1:, col_idx]  # skip header

            if col_name in currency_cols:
                col_range.number_format = '#,##0.00'  # e.g., 1,234,567.89
                print(f"Formatted '{col_name}' as currency")

            elif col_name in percent_cols:
                col_range.number_format = '0.00%'     # e.g., 68.27%
                print(f"Formatted '{col_name}' as percent")

            elif col_name in round_cols:
                col_range.number_format = '0.00'      # plain float
                print(f"Formatted '{col_name}' as float")

        # Autofit everything
        sht.autofit()
        print("Formatting complete ✅")

    except Exception as e:
        print("❌ Formatting failed:", str(e))


@script
def score_credit_card_segment(book: xw.Book):
    print("Starting score_credit_card_segment...")

    try:
        sht = book.sheets.active
        table = sht.tables['RBICC']
        rng = table.range
        print(f"Loaded table 'RBICC' from sheet: {sht.name}")

        df = table.range.options(pd.DataFrame, index=False).value
        print("Loaded table into DataFrame.")

        # Ensure correct type for scoring
        for col in df.columns:
            try:
                df[col] = df[col].astype(float)
            except:
                continue

        print("Converted numeric columns to float where possible.")

        # Scoring variables and weights
        features = {
            'TOTAL_CC_TXN_VOLUME_NOS': 20,
            'TOTAL_CC_TXN_VALUE_AMT': 20,
            'AVG_CC_TICKET_SIZE': 10,
            'POS_SHARE_OF_CC_VOLUME': 10,
            'ECOM_SHARE_OF_CC_VOLUME': 10,
            'CC_TO_DC_TXN_RATIO': 15,
            'CC_TO_DC_VALUE_RATIO': 15,
        }

        score = pd.Series(0.0, index=df.index)

        for col, weight in features.items():
            if col not in df.columns:
                print(f"Missing column for scoring: {col}")
                continue

            col_min = df[col].min()
            col_max = df[col].max()
            print(f"Normalizing {col} (min={col_min}, max={col_max})")

            # Avoid divide-by-zero
            if col_max - col_min == 0:
                normalized = 0
            else:
                normalized = (df[col] - col_min) / (col_max - col_min)

            score += normalized * weight

        df['CREDIT_CARD_SCORE'] = score.round(2)
        print("Scoring complete. Added 'CREDIT_CARD_SCORE' column.")

        # Update the table in place with new column
        table.range.value = df
        print("Updated table with score column.")

        sht.autofit()
        print("✅ Score calculation and insertion complete.")

    except Exception as e:
        print("❌ Error during scoring:", str(e))


@script
def generate_multicollinearity_matrix(book: xw.Book):
    print("Starting generate_multicollinearity_matrix...")

    try:
        sht = book.sheets.active
        table = sht.tables['RBICC']
        print(f"Loaded table 'RBICC' from sheet: {sht.name}")

        # Load data
        df = table.range.options(pd.DataFrame, index=False).value
        print("Table loaded into DataFrame.")

        # Keep only numeric columns
        numeric_df = df.select_dtypes(include='number')
        print(f"Selected {len(numeric_df.columns)} numeric columns.")

        # Calculate correlation matrix
        corr_matrix = numeric_df.corr().round(2)
        print("Correlation matrix calculated.")

        # Write to new sheet
        if 'RBICC_CorrMatrix' in [s.name for s in book.sheets]:
            book.sheets['RBICC_CorrMatrix'].delete()

        corr_sht = book.sheets.add('RBICC_CorrMatrix')
        corr_sht.range("A1").value = corr_matrix
        corr_sht.autofit()

        print("✅ Correlation matrix inserted into 'RBICC_CorrMatrix'.")

    except Exception as e:
        print("❌ Error during correlation matrix generation:", str(e))
```

### 4. Database Integration
This script demonstrates a modular approach to connecting to a database API. It reads connection details and parameters from a `MASTER` sheet, executes a query, and outputs the results to a new sheet. This single function can be adapted for various database operations.

```python
import xlwings as xw
import pandas as pd
from pyodide.http import pyfetch
import urllib.parse
import js

async def _execute_query(api_url, params):
    """Helper function to execute a query against the API."""
    query_string = urllib.parse.urlencode(params)
    full_url = f"{api_url}?{query_string}"
    
    # Standard logging placeholder: Consider adding logging for the full_url in a debug mode.
    
    response = await pyfetch(
        full_url,
        method="GET",
        headers={"Accept": "text/plain,application/json"},
        response_type="blob"
    )
    
    if not response.ok:
        raise Exception(f"API Error: {response.status} - {await response.text()}")
        
    return await response.text()

def _parse_pipe_delimited(text_content):
    """Helper function to parse pipe-delimited text into a DataFrame."""
    lines = text_content.strip().split("
")
    if not lines or not lines[0]:
        return pd.DataFrame()
        
    headers = [h.strip() for h in lines[0].split("|")]
    data_rows = [
        [cell.strip() for cell in line.split("|")]
        for line in lines[1:] if line.strip()
    ]
    
    df = pd.DataFrame(data_rows, columns=headers)
    
    # Attempt to convert columns to numeric types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
        
    return df

@script
async def run_db_query(book: xw.Book, action: str):
    """
    Connects to a database API and performs an action based on user input.
    
    Args:
        book (xw.Book): The workbook object.
        action (str): The action to perform. One of: 
                      'list_tables', 'get_table_data', 'get_random_records'.
    """
    try:
        # 1. Read connection details and parameters from a MASTER sheet
        master_sheet = book.sheets["MASTER"]
        api_url = master_sheet["B2"].value
        
        connection_params = {
            "host": master_sheet["B3"].value,
            "database": master_sheet["B4"].value,
            "user": master_sheet["B5"].value,
            "password": master_sheet["B6"].value,
            "port": int(master_sheet["B7"].value) if master_sheet["B7"].value else 5432,
            "db_type": master_sheet["B8"].value,
        }
        
        schema = master_sheet["B9"].value or "public"
        table_name = master_sheet["B11"].value
        num_records = int(master_sheet["B12"].value) if master_sheet["B12"].value else 100
        
        # 2. Build the SQL query based on the specified action
        sql_query = ""
        if action == 'list_tables':
            # SQL to list all tables (example for PostgreSQL)
            sql_query = f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}' ORDER BY table_name;"
        elif action == 'get_table_data':
            if not table_name:
                raise ValueError("Table name is required for 'get_table_data' action.")
            # SQL to get the first N records
            sql_query = f"SELECT * FROM {schema}.{table_name} LIMIT {num_records};"
        elif action == 'get_random_records':
            if not table_name:
                raise ValueError("Table name is required for 'get_random_records' action.")
            # SQL to get N random records (example for PostgreSQL)
            sql_query = f"SELECT * FROM {schema}.{table_name} ORDER BY RANDOM() LIMIT {num_records};"
        else:
            raise ValueError(f"Invalid action: {action}")

        connection_params["sqlquery"] = sql_query
        
        # 3. Execute the query using the API
        response_text = await _execute_query(api_url, connection_params)
        
        # 4. Parse the response into a DataFrame
        df = _parse_pipe_delimited(response_text)
        
        # 5. Write the DataFrame to a new Excel sheet
        sheet_name = f"DB_{action.upper()}"
        if table_name:
            sheet_name += f"_{table_name.upper()}"
        sheet_name = sheet_name[:31] # Enforce Excel's sheet name length limit

        if sheet_name in [s.name for s in book.sheets]:
            book.sheets[sheet_name].delete()
            
        sheet = book.sheets.add(name=sheet_name)
        
        # 6. Place the data and format it as a table
        sheet["A1"].value = df
        
        try:
            # This demonstrates how to format the output as an Excel table
            table_range = sheet["A1"].expand()
            sheet.tables.add(source=table_range, name=f"tbl_{sheet_name}")
            # Standard logging placeholder: Add confirmation message for table creation.
        except Exception as e:
            # Standard logging placeholder: Add warning if table formatting fails.
            pass # Fail gracefully if table creation is not supported

        print(f"✅ Successfully executed '{action}' and updated sheet '{sheet_name}'.")

    except Exception as e:
        # Standard logging placeholder: Replace with robust error handling and user feedback.
        print(f"❌ An error occurred: {e}")

# To use this script, you would call it from another function or a button, like so:
# @script
# async def list_all_tables(book: xw.Book):
#     await run_db_query(book, 'list_tables')
# 
# @script
# async def get_sample_data(book: xw.Book):
#     await run_db_query(book, 'get_table_data')

```

### 5. Web Scraping with LLM Processing (URL_LIST.xlsx)
This script demonstrates a more complex workflow:
- Reading a list of URLs from an Excel table
- Scraping the content of each URL using an external API (Firecrawl)
- Processing the scraped content with a Large Language Model (LLM)
- Writing the results back to a new sheet in Excel

```python
@script
async def scrape_and_process(book: xw.Book):
    print("🚀 Starting URL scraping and processing...")

    # 1. Read configuration from MASTER sheet
    try:
        master_sheet = book.sheets["MASTER"]
        api_url = master_sheet["B2"].value
        llm_provider = master_sheet["B3"].value
        # Add other parameters as needed
    except Exception as e:
        print(f"❌ Error reading configuration from MASTER sheet: {e}")
        return

    # 2. Read list of URLs from the active sheet
    try:
        sht = book.sheets.active
        url_table = sht.tables['URL_LIST']
        urls = url_table.range.options(pd.DataFrame, index=False).value
        print(f"Found {len(urls)} URLs to process.")
    except Exception as e:
        print(f"❌ Error reading URL list from table 'URL_LIST': {e}")
        return

    # 3. Process each URL
    results = []
    for index, row in urls.iterrows():
        url = row['URL']
        print(f"Processing URL: {url}")
        
        try:
            # a. Scrape the URL content via Firecrawl API
            # This would be an async call to your own API wrapper for Firecrawl
            scraped_data = await your_firecrawl_wrapper(api_url, url)
            
            # b. Process the content with an LLM
            # This would be another async call to your LLM API wrapper
            processed_content = await your_llm_wrapper(llm_provider, scraped_data)
            
            results.append({
                "URL": url,
                "Scraped_Content": scraped_data,
                "LLM_Summary": processed_content
            })
            print(f"✅ Successfully processed {url}")

        except Exception as e:
            print(f"⚠️ Failed to process {url}: {e}")
            results.append({
                "URL": url,
                "Scraped_Content": "Error",
                "LLM_Summary": str(e)
            })

    # 4. Write results to a new sheet
    if results:
        results_df = pd.DataFrame(results)
        sheet_name = "Scraping_Results"
        if sheet_name in [s.name for s in book.sheets]:
            book.sheets[sheet_name].delete()
        
        new_sheet = book.sheets.add(name=sheet_name)
        new_sheet["A1"].options(index=False).value = results_df
        
        try:
            new_sheet.tables.add(source=new_sheet["A1"].expand())
        except Exception as e:
            print(f"⚠️ Could not format results as a table: {e}")
            
        print(f"✅ Finished processing. Results are in the '{sheet_name}' sheet.")

```

### 6. Advanced EDA and Schema Analysis with LLMs
This script demonstrates how to perform an Exploratory Data Analysis (EDA) and schema detection on a given table using an LLM.

```python
@script
async def analyze_table_with_llm(book: xw.Book):
    print("🤖 Starting table analysis with LLM...")

    # 1. Read configuration from MASTER sheet
    try:
        master_sheet = book.sheets["MASTER"]
        llm_provider = master_sheet["B3"].value
        table_name = master_sheet["B11"].value
    except Exception as e:
        print(f"❌ Error reading configuration from MASTER sheet: {e}")
        return

    # 2. Get the data from the specified table
    try:
        sht = book.sheets.active
        data_table = sht.tables[table_name]
        df = data_table.range.options(pd.DataFrame, index=False).value
        print(f"Loaded table '{table_name}' with shape {df.shape}")
    except Exception as e:
        print(f"❌ Could not read table '{table_name}': {e}")
        return

    # 3. Prepare the data and prompt for the LLM
    # For this example, we'll send the first 5 rows as a CSV string
    data_sample = df.head(5).to_csv(index=False)
    
    prompt = f"""
    Analyze the following table data and provide a summary of its schema and potential insights.
    
    Data Sample:
    {data_sample}
    
    Please provide:
    1. A description of each column, including its likely data type and purpose.
    2. A summary of the overall dataset.
    3. Three potential business questions that could be answered using this data.
    """

    # 4. Call the LLM API
    try:
        # This would be an async call to your LLM API wrapper
        llm_response = await your_llm_wrapper(llm_provider, prompt)
        print("✅ LLM analysis complete.")
    except Exception as e:
        print(f"❌ LLM API call failed: {e}")
        return

    # 5. Write the LLM response to a new sheet
    sheet_name = f"LLM_Analysis_{table_name}"
    sheet_name = sheet_name[:31]

    if sheet_name in [s.name for s in book.sheets]:
        book.sheets[sheet_name].delete()
        
    new_sheet = book.sheets.add(name=sheet_name)
    new_sheet["A1"].value = "LLM Analysis"
    new_sheet["A2"].value = llm_response
    
    print(f"✅ Analysis complete. Results are in the '{sheet_name}' sheet.")

```