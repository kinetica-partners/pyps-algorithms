"""
Pytest test for AST consistency between main and xlwings lite modules.

This test ensures that the core algorithmic logic remains identical
between main and lite implementations without requiring runtime imports
of xlwings modules.
"""

import pytest
from pathlib import Path
from typing import Dict, Any

try:
    from .helpers.ast_comparison import compare_modules, print_comparison_report
except ImportError:
    # Fallback for direct execution
    import sys
    helpers_path = Path(__file__).parent / "helpers"
    sys.path.insert(0, str(helpers_path))
    from ast_comparison import compare_modules, print_comparison_report  # type: ignore


class TestASTConsistency:
    """Test AST consistency between main and xlwings lite modules."""
    
    def test_working_calendar_consistency_when_available(self):
        """Test consistency for working_calendar modules when xlwings lite is available."""
        main_file = Path("src/working_calendar.py")
        lite_file = Path("xlwings_lite/working_calendar_lite.py")  # Future location
        
        # Skip if either file doesn't exist
        if not main_file.exists() or not lite_file.exists():
            pytest.skip("Working calendar files not found")
        
        results = compare_modules(main_file, lite_file, normalize_names=False)
        
        # Print detailed report
        print_comparison_report(results)
        
        # Define core algorithmic functions that MUST be identical
        core_functions = {
            'convert_excel_time',
            'convert_excel_boolean',
            'parse_time_string',
            'load_calendar_rules',
            'build_working_intervals',
            'add_working_minutes'
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
        lite_file = Path("xlwings_lite/explode_bom_lite.py")  # Future location
        
        # Skip if either file doesn't exist
        if not main_file.exists() or not lite_file.exists():
            pytest.skip("Explode BOM files not found")
        
        results = compare_modules(main_file, lite_file, normalize_names=False)
        
        # Print detailed report
        print_comparison_report(results)
        
        # Define core algorithmic functions for BOM explosion
        core_functions = {
            'explode_bom_iterative'
        }
        
        # Check consistency
        existing_core_functions = core_functions & set(results['comparisons'].keys())
        identical_core_functions = existing_core_functions & set(results['identical'])
        
        # All existing core functions should be identical
        assert identical_core_functions == existing_core_functions, \
            f"Core BOM functions should be identical. Differences found in: {existing_core_functions - identical_core_functions}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])