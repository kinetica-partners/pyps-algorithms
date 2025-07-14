#!/usr/bin/env python3
"""
Test suite for BOM explosion functionality.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from explode_bom import (
    explode_bom_iterative,
    bom_explosion_from_csv,
    bom_explosion_from_excel
)
from config import get_file_path, get_test_data_path


class TestBOMExplosion:
    """Test BOM explosion functionality."""
    
    @pytest.fixture
    def sample_bom_data(self):
        """Create sample BOM data for testing."""
        independent_demand = [
            {'item': 'PROD_A', 'quantity': 10, 'due_date': '2025-01-15'},
            {'item': 'PROD_B', 'quantity': 5, 'due_date': '2025-01-20'},
        ]
        
        items = [
            {'item': 'PROD_A', 'production_lead_time': 2},
            {'item': 'PROD_B', 'production_lead_time': 3},
            {'item': 'COMP_X', 'production_lead_time': 1},
            {'item': 'COMP_Y', 'production_lead_time': 1},
            {'item': 'RAW_M', 'production_lead_time': 0},
        ]
        
        bom = [
            {'parent_item': 'PROD_A', 'child_item': 'COMP_X', 'quantity_per': 2},
            {'parent_item': 'PROD_A', 'child_item': 'RAW_M', 'quantity_per': 1},
            {'parent_item': 'PROD_B', 'child_item': 'COMP_Y', 'quantity_per': 1},
            {'parent_item': 'COMP_X', 'child_item': 'RAW_M', 'quantity_per': 3},
        ]
        
        return independent_demand, items, bom
    
    def test_explode_bom_iterative_basic(self, sample_bom_data):
        """Test basic BOM explosion functionality."""
        independent_demand, items, bom = sample_bom_data
        
        result = explode_bom_iterative(independent_demand, items, bom)
        
        # Should return a list of demand records
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Check that we have the expected structure
        for record in result:
            assert 'product_item' in record
            assert 'quantity' in record
            assert 'due_date' in record
            assert 'parent_item' in record
            assert 'child_item' in record
            assert 'child_qty' in record
            assert 'child_due_date' in record
    
    def test_explode_bom_quantities(self, sample_bom_data):
        """Test that quantities are correctly calculated."""
        independent_demand, items, bom = sample_bom_data
        
        result = explode_bom_iterative(independent_demand, items, bom)
        
        # Find demand for COMP_X from PROD_A (should be 10 * 2 = 20)
        comp_x_demand = [r for r in result if r['child_item'] == 'COMP_X' and r['parent_item'] == 'PROD_A']
        assert len(comp_x_demand) == 1
        assert comp_x_demand[0]['child_qty'] == 20
        
        # Find demand for RAW_M from COMP_X (should be 20 * 3 = 60)
        raw_m_from_comp_x = [r for r in result if r['child_item'] == 'RAW_M' and r['parent_item'] == 'COMP_X']
        assert len(raw_m_from_comp_x) == 1
        assert raw_m_from_comp_x[0]['child_qty'] == 60
    
    def test_explode_bom_dates(self, sample_bom_data):
        """Test that dates are correctly calculated with lead times."""
        independent_demand, items, bom = sample_bom_data
        
        result = explode_bom_iterative(independent_demand, items, bom)
        
        # Find demand for COMP_X from PROD_A
        comp_x_demand = [r for r in result if r['child_item'] == 'COMP_X' and r['parent_item'] == 'PROD_A']
        assert len(comp_x_demand) == 1
        
        # Due date should be pushed back by PROD_A's lead time (2 days)
        # Original due date: 2025-01-15, less 2 days = 2025-01-13
        expected_date = '2025-01-13'
        assert comp_x_demand[0]['child_due_date'] == expected_date
    
    def test_explode_bom_multiple_products(self, sample_bom_data):
        """Test BOM explosion with multiple products."""
        independent_demand, items, bom = sample_bom_data
        
        result = explode_bom_iterative(independent_demand, items, bom)
        
        # Should have demand from both PROD_A and PROD_B
        product_a_records = [r for r in result if r['product_item'] == 'PROD_A']
        product_b_records = [r for r in result if r['product_item'] == 'PROD_B']
        
        assert len(product_a_records) > 0
        assert len(product_b_records) > 0
    
    def test_explode_bom_empty_input(self):
        """Test BOM explosion with empty input."""
        result = explode_bom_iterative([], [], [])
        assert result == []
    
    def test_explode_bom_no_bom_structure(self):
        """Test with independent demand but no BOM structure."""
        independent_demand = [{'item': 'SIMPLE', 'quantity': 1, 'due_date': '2025-01-15'}]
        items = [{'item': 'SIMPLE', 'production_lead_time': 0}]
        bom = []
        
        result = explode_bom_iterative(independent_demand, items, bom)
        
        # Should return empty result since no BOM structure exists
        assert result == []
    
    def test_bom_explosion_from_csv_integration(self):
        """Test CSV-based BOM explosion integration."""
        try:
            # Try to run with actual data files
            result = bom_explosion_from_csv(dataset='current')
            
            # Should return a success message
            assert isinstance(result, str)
            assert 'successfully' in result.lower()
            
        except FileNotFoundError:
            # Skip test if data files don't exist
            pytest.skip("BOM data files not found in ./data/current")
        except Exception as e:
            pytest.fail(f"BOM explosion from CSV failed: {e}")
    
    def test_bom_explosion_csv_file_creation(self):
        """Test that CSV explosion creates the expected output file."""
        try:
            # Run the explosion
            result = bom_explosion_from_csv(dataset='current')
            
            # Check that output file was created
            output_file = get_file_path('current', 'total_demand')
            assert output_file.exists(), "Output file was not created"
            
            # Check that the output file has content
            output_df = pd.read_csv(output_file)
            assert not output_df.empty, "Output file is empty"
            
            # Check expected columns
            expected_columns = ['product_item', 'quantity', 'due_date', 'parent_item', 'child_item', 'child_qty', 'child_due_date']
            for col in expected_columns:
                assert col in output_df.columns, f"Missing column: {col}"
                
        except FileNotFoundError:
            pytest.skip("BOM data files not found in ./data/current")
        except Exception as e:
            pytest.fail(f"BOM explosion CSV test failed: {e}")
    
    def test_bom_explosion_date_formats(self):
        """Test BOM explosion with different date formats."""
        independent_demand = [
            {'item': 'TEST', 'quantity': 1, 'due_date': datetime(2025, 1, 15)},  # datetime object
            {'item': 'TEST2', 'quantity': 1, 'due_date': '2025-01-20'},  # string
        ]
        
        items = [
            {'item': 'TEST', 'production_lead_time': 1},
            {'item': 'TEST2', 'production_lead_time': 2},
            {'item': 'CHILD', 'production_lead_time': 0},
        ]
        
        bom = [
            {'parent_item': 'TEST', 'child_item': 'CHILD', 'quantity_per': 1},
            {'parent_item': 'TEST2', 'child_item': 'CHILD', 'quantity_per': 1},
        ]
        
        result = explode_bom_iterative(independent_demand, items, bom)
        
        # Should handle both datetime and string formats
        assert len(result) == 2
        
        # Check that dates are properly formatted in output
        for record in result:
            assert isinstance(record['due_date'], str)
            assert isinstance(record['child_due_date'], str)
    
    def test_bom_explosion_lead_time_calculation(self):
        """Test detailed lead time calculations."""
        independent_demand = [
            {'item': 'PARENT', 'quantity': 1, 'due_date': '2025-01-15'},
        ]
        
        items = [
            {'item': 'PARENT', 'production_lead_time': 3},
            {'item': 'CHILD', 'production_lead_time': 2},
        ]
        
        bom = [
            {'parent_item': 'PARENT', 'child_item': 'CHILD', 'quantity_per': 1},
        ]
        
        result = explode_bom_iterative(independent_demand, items, bom)
        
        assert len(result) == 1
        record = result[0]
        
        # Child due date should be parent due date minus parent lead time
        # 2025-01-15 - 3 days = 2025-01-12
        assert record['child_due_date'] == '2025-01-12'
        assert record['due_date'] == '2025-01-15'  # Original due date preserved


class TestBOMExcelIntegration:
    """Test Excel integration functionality."""
    
    def test_bom_explosion_from_excel_function_exists(self):
        """Test that Excel function exists and is callable."""
        # This is a basic test to ensure the function exists
        assert callable(bom_explosion_from_excel)
    
    def test_bom_explosion_from_excel_without_workbook(self):
        """Test Excel function behavior when no workbook is available."""
        # This test will likely fail in a non-Excel environment
        # but we test that it fails gracefully
        try:
            result = bom_explosion_from_excel()
            # If it succeeds, it should return a string result
            assert isinstance(result, str)
        except Exception as e:
            # Expected to fail without Excel/xlwings environment
            assert "xlwings" in str(e).lower() or "excel" in str(e).lower() or "workbook" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__])