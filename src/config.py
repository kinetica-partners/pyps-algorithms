"""
Configuration module for PyPS Scheduling Algorithms
Provides cross-platform path resolution using pathlib.Path
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class PathConfig:
    """Cross-platform path configuration manager"""
    
    def __init__(self, config_file: str = "config.yaml"):
        """Initialize configuration from YAML file"""
        self._project_root = self._find_project_root()
        self._config_path = self._project_root / config_file
        self._config = self._load_config()
    
    def _find_project_root(self) -> Path:
        """Find project root directory by looking for key files"""
        current = Path(__file__).parent
        
        # Look for project root indicators
        root_indicators = ["config.yaml", "pyproject.toml", "README.md", "main.py"]
        
        while current != current.parent:
            if any((current / indicator).exists() for indicator in root_indicators):
                return current
            current = current.parent
        
        # If not found, use parent of src directory
        return Path(__file__).parent.parent
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self._config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    @property
    def project_root(self) -> Path:
        """Get project root directory"""
        return self._project_root
    
    def get_path(self, *path_parts: str) -> Path:
        """Get absolute path from project root"""
        return self._project_root / Path(*path_parts)
    
    def get_data_path(self, dataset: str = "current") -> Path:
        """Get data directory path"""
        if dataset == "current":
            return self.get_path(self._config["data"]["current"])
        elif dataset == "simple":
            return self.get_path(self._config["data"]["simple"])
        elif dataset == "test":
            return self.get_path(self._config["data"]["test"])
        elif dataset == "rnc":
            return self.get_path(self._config["data"]["rnc"])
        elif dataset == "xpreso":
            return self.get_path(self._config["data"]["xpreso"])
        elif dataset == "twelve_pyr":
            return self.get_path(self._config["data"]["twelve_pyr"])
        else:
            return self.get_path(self._config["data"]["root"])
    
    def get_file_path(self, dataset: str, file_type: str) -> Path:
        """Get specific file path within a dataset"""
        data_path = self.get_data_path(dataset)
        filename = self._config["files"][file_type]
        return data_path / filename
    
    def get_test_data_path(self, filename: Optional[str] = None) -> Path:
        """Get test data directory or specific test file"""
        test_data_path = self.get_path(self._config["test"]["data"])
        if filename:
            return test_data_path / filename
        return test_data_path
    
    def get_test_file_path(self, file_type: str) -> Path:
        """Get specific test file path"""
        return self.get_path(self._config["test_files"][file_type])
    
    def get_src_path(self, filename: Optional[str] = None) -> Path:
        """Get source directory or specific source file"""
        src_path = self.get_path(self._config["src"]["root"])
        if filename:
            return src_path / filename
        return src_path
    
    def get_docs_path(self, filename: Optional[str] = None) -> Path:
        """Get docs directory or specific docs file"""
        docs_path = self.get_path(self._config["docs"]["root"])
        if filename:
            return docs_path / filename
        return docs_path
    
    def ensure_dir_exists(self, path: Path) -> Path:
        """Ensure directory exists, create if not"""
        path.mkdir(parents=True, exist_ok=True)
        return path


# Global configuration instance
_config = None

def get_config() -> PathConfig:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = PathConfig()
    return _config


# Convenience functions for common operations
def get_project_root() -> Path:
    """Get project root directory"""
    return get_config().project_root

def get_data_path(dataset: str = "current") -> Path:
    """Get data directory path"""
    return get_config().get_data_path(dataset)

def get_file_path(dataset: str, file_type: str) -> Path:
    """Get specific file path within a dataset"""
    return get_config().get_file_path(dataset, file_type)

def get_test_data_path(filename: Optional[str] = None) -> Path:
    """Get test data directory or specific test file"""
    return get_config().get_test_data_path(filename)

def get_src_path(filename: Optional[str] = None) -> Path:
    """Get source directory or specific source file"""
    return get_config().get_src_path(filename)


if __name__ == "__main__":
    # Demo usage
    config = get_config()
    print(f"Project root: {config.project_root}")
    print(f"Current data path: {get_data_path('current')}")
    print(f"BOM file path: {get_file_path('current', 'bom')}")
    print(f"Test data path: {get_test_data_path()}")
    print(f"Source path: {get_src_path()}")