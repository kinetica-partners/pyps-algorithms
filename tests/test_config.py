#!/usr/bin/env python3
"""
Comprehensive test suite for the portable configuration system.
Tests real functionality with actual files to improve coverage.
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path
import yaml

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import PathConfig, get_file_path, get_data_path, get_project_root, get_test_data_path, get_src_path


class TestPathConfigCore:
    """Test core PathConfig functionality."""
    
    def test_init_with_actual_config(self):
        """Test PathConfig initialization with actual configuration."""
        config = PathConfig()
        assert config.project_root is not None
        assert config._config is not None
        assert isinstance(config.project_root, Path)
        assert config._config_path.exists()
    
    def test_project_root_property(self):
        """Test project_root property access."""
        config = PathConfig()
        root = config.project_root
        assert isinstance(root, Path)
        assert root.exists()
        # Should contain key project files
        expected_files = ['config.yaml', 'pyproject.toml', 'README.md']
        assert any((root / file).exists() for file in expected_files)
    
    def test_get_path_method(self):
        """Test get_path method with various inputs."""
        config = PathConfig()
        
        # Test single path component
        path = config.get_path('data')
        assert isinstance(path, Path)
        assert path.is_absolute()
        
        # Test multiple path components
        path = config.get_path('data', 'current', 'bom.csv')
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert str(path).endswith('bom.csv')


class TestDataPathResolution:
    """Test data path resolution functionality."""
    
    def test_get_data_path_current(self):
        """Test getting current data directory path."""
        config = PathConfig()
        path = config.get_data_path('current')
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert 'current' in str(path)
    
    def test_get_data_path_test(self):
        """Test getting test data directory path."""
        config = PathConfig()
        path = config.get_data_path('test')
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert 'test' in str(path)
    
    def test_get_data_path_simple(self):
        """Test getting simple data directory path."""
        config = PathConfig()
        path = config.get_data_path('simple')
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert 'simple' in str(path)
    
    def test_get_data_path_rnc(self):
        """Test getting RnC data directory path."""
        config = PathConfig()
        path = config.get_data_path('rnc')
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert 'RnC' in str(path)
    
    def test_get_data_path_xpreso(self):
        """Test getting xpreso data directory path."""
        config = PathConfig()
        path = config.get_data_path('xpreso')
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert 'xpreso' in str(path)
    
    def test_get_data_path_twelve_pyr(self):
        """Test getting 12PYR data directory path."""
        config = PathConfig()
        path = config.get_data_path('twelve_pyr')
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert '12PYR' in str(path)
    
    def test_get_data_path_default_fallback(self):
        """Test getting data path with unknown dataset falls back to root."""
        config = PathConfig()
        path = config.get_data_path('unknown_dataset')
        assert isinstance(path, Path)
        assert path.is_absolute()


class TestFilePathResolution:
    """Test file path resolution functionality."""
    
    def test_get_file_path_bom(self):
        """Test getting BOM file path."""
        config = PathConfig()
        path = config.get_file_path('current', 'bom')
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert str(path).endswith('bom.csv')
        assert 'current' in str(path)
    
    def test_get_file_path_items(self):
        """Test getting items file path."""
        config = PathConfig()
        path = config.get_file_path('current', 'items')
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert str(path).endswith('items.csv')
    
    def test_get_file_path_independent_demand(self):
        """Test getting independent demand file path."""
        config = PathConfig()
        path = config.get_file_path('current', 'independent_demand')
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert str(path).endswith('independent_demand.csv')
    
    def test_get_file_path_calendar_rules(self):
        """Test getting calendar rules file path."""
        config = PathConfig()
        path = config.get_file_path('current', 'calendar_rules')
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert str(path).endswith('calendar_rules.csv')
    
    def test_get_file_path_calendar_exceptions(self):
        """Test getting calendar exceptions file path."""
        config = PathConfig()
        path = config.get_file_path('current', 'calendar_exceptions')
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert str(path).endswith('calendar_exceptions.csv')


class TestTestDataPaths:
    """Test test data path functionality."""
    
    def test_get_test_data_path_directory(self):
        """Test getting test data directory."""
        config = PathConfig()
        path = config.get_test_data_path()
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert 'test_data' in str(path)
    
    def test_get_test_data_path_with_filename(self):
        """Test getting specific test data file."""
        config = PathConfig()
        path = config.get_test_data_path('baseline.json')
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert str(path).endswith('baseline.json')
        assert 'test_data' in str(path)
    
    def test_get_test_file_path_baseline(self):
        """Test getting test baseline file path."""
        config = PathConfig()
        path = config.get_test_file_path('baseline')
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert str(path).endswith('baseline.json')


class TestSourcePaths:
    """Test source path functionality."""
    
    def test_get_src_path_directory(self):
        """Test getting source directory."""
        config = PathConfig()
        path = config.get_src_path()
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert str(path).endswith('src')
    
    def test_get_src_path_with_filename(self):
        """Test getting specific source file."""
        config = PathConfig()
        path = config.get_src_path('config.py')
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert str(path).endswith('config.py')
        assert 'src' in str(path)


class TestDocsPaths:
    """Test docs path functionality."""
    
    def test_get_docs_path_directory(self):
        """Test getting docs directory."""
        config = PathConfig()
        path = config.get_docs_path()
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert str(path).endswith('docs')
    
    def test_get_docs_path_with_filename(self):
        """Test getting specific docs file."""
        config = PathConfig()
        path = config.get_docs_path('README.md')
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert str(path).endswith('README.md')


class TestUtilityMethods:
    """Test utility methods."""
    
    def test_ensure_dir_exists_new_directory(self):
        """Test ensure_dir_exists creates new directory."""
        config = PathConfig()
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / 'new_directory'
            assert not test_path.exists()
            
            result = config.ensure_dir_exists(test_path)
            assert test_path.exists()
            assert test_path.is_dir()
            assert result == test_path
    
    def test_ensure_dir_exists_existing_directory(self):
        """Test ensure_dir_exists with existing directory."""
        config = PathConfig()
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir)
            assert test_path.exists()
            
            result = config.ensure_dir_exists(test_path)
            assert test_path.exists()
            assert result == test_path


class TestConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_get_project_root_function(self):
        """Test get_project_root convenience function."""
        root = get_project_root()
        assert isinstance(root, Path)
        assert root.exists()
    
    def test_get_data_path_function(self):
        """Test get_data_path convenience function."""
        path = get_data_path('current')
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert 'current' in str(path)
    
    def test_get_file_path_function(self):
        """Test get_file_path convenience function."""
        path = get_file_path('current', 'bom')
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert str(path).endswith('bom.csv')
    
    def test_get_test_data_path_function(self):
        """Test get_test_data_path convenience function."""
        path = get_test_data_path()
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert 'test_data' in str(path)
    
    def test_get_test_data_path_function_with_filename(self):
        """Test get_test_data_path convenience function with filename."""
        path = get_test_data_path('baseline.json')
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert str(path).endswith('baseline.json')
    
    def test_get_src_path_function(self):
        """Test get_src_path convenience function."""
        path = get_src_path()
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert str(path).endswith('src')
    
    def test_get_src_path_function_with_filename(self):
        """Test get_src_path convenience function with filename."""
        path = get_src_path('config.py')
        assert isinstance(path, Path)
        assert path.is_absolute()
        assert str(path).endswith('config.py')


class TestActualFileAccess:
    """Test accessing actual project files."""
    
    def test_current_data_files_exist(self):
        """Test that current data files exist."""
        config = PathConfig()
        
        # Test files that should exist
        bom_path = config.get_file_path('current', 'bom')
        assert bom_path.exists(), f"BOM file should exist at {bom_path}"
        
        items_path = config.get_file_path('current', 'items')
        assert items_path.exists(), f"Items file should exist at {items_path}"
        
        demand_path = config.get_file_path('current', 'independent_demand')
        assert demand_path.exists(), f"Demand file should exist at {demand_path}"
    
    def test_test_data_files_exist(self):
        """Test that test data files exist."""
        config = PathConfig()
        
        baseline_path = config.get_test_file_path('baseline')
        assert baseline_path.exists(), f"Baseline file should exist at {baseline_path}"
        
        test_dir = config.get_test_data_path()
        assert test_dir.exists(), f"Test data directory should exist at {test_dir}"
    
    def test_source_files_exist(self):
        """Test that source files exist."""
        config = PathConfig()
        
        config_file = config.get_src_path('config.py')
        assert config_file.exists(), f"Config file should exist at {config_file}"
        
        src_dir = config.get_src_path()
        assert src_dir.exists(), f"Source directory should exist at {src_dir}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])