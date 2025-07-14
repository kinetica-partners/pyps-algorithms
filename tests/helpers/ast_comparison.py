"""
AST Comparison Tool for xlwings lite module consistency checking.

This tool compares the algorithmic structure of Python functions while ignoring:
- Import statements  
- Decorators (@xw.func, etc.)
- Comments
- Variable names (optional)
- Whitespace and formatting
"""

import ast
import sys
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path


class FunctionExtractor(ast.NodeVisitor):
    """Extract function definitions from an AST, ignoring decorators."""
    
    def __init__(self):
        self.functions = {}
    
    def visit_FunctionDef(self, node):
        """Extract function definition without decorators."""
        # Store a copy of the node without decorators
        import copy
        clean_func = copy.deepcopy(node)
        clean_func.decorator_list = []  # Remove decorators
        
        self.functions[node.name] = clean_func
        self.generic_visit(node)


class ASTNormalizer(ast.NodeTransformer):
    """Normalize AST nodes for consistent comparison."""
    
    def __init__(self, normalize_names=False):
        self.normalize_names = normalize_names
        self.name_mapping = {}
        self.counter = 0
    
    def visit_Name(self, node):
        """Optionally normalize variable names for comparison."""
        if self.normalize_names and isinstance(node.ctx, (ast.Load, ast.Store)):
            if node.id not in self.name_mapping:
                self.name_mapping[node.id] = f"var_{self.counter}"
                self.counter += 1
            node.id = self.name_mapping[node.id]
        return node
    
    def visit_arg(self, node):
        """Normalize function argument names."""
        if self.normalize_names:
            if node.arg not in self.name_mapping:
                self.name_mapping[node.arg] = f"arg_{len(self.name_mapping)}"
            node.arg = self.name_mapping[node.arg]
        return node


def extract_functions_from_file(file_path: Path) -> Dict[str, ast.FunctionDef]:
    """Extract function definitions from a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Parse the source into an AST
        tree = ast.parse(source)
        
        # Extract functions
        extractor = FunctionExtractor()
        extractor.visit(tree)
        
        return extractor.functions
    
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return {}


def normalize_function(func_node: ast.FunctionDef, normalize_names=False) -> str:
    """Convert function AST to normalized string representation."""
    normalizer = ASTNormalizer(normalize_names)
    normalized = normalizer.visit(func_node)
    return ast.dump(normalized, indent=2)


def compare_functions(func1: ast.FunctionDef, func2: ast.FunctionDef, 
                     normalize_names=False) -> Tuple[bool, List[str]]:
    """Compare two function ASTs and return differences."""
    differences = []
    
    # Compare function signatures
    if func1.name != func2.name:
        differences.append(f"Function name: '{func1.name}' vs '{func2.name}'")
    
    # Compare argument count
    args1 = len(func1.args.args)
    args2 = len(func2.args.args)
    if args1 != args2:
        differences.append(f"Argument count: {args1} vs {args2}")
    
    # Compare function bodies (normalized)
    body1 = normalize_function(func1, normalize_names)
    body2 = normalize_function(func2, normalize_names)
    
    if body1 != body2:
        differences.append("Function body structure differs")
        # For debugging, you could add more detailed diff here
    
    return len(differences) == 0, differences


def compare_modules(file1: Path, file2: Path, normalize_names=False) -> Dict[str, Any]:
    """Compare two Python modules and return detailed comparison results."""
    print(f"Comparing {file1.name} vs {file2.name}")
    
    # Extract functions from both files
    functions1 = extract_functions_from_file(file1)
    functions2 = extract_functions_from_file(file2)
    
    results = {
        'file1': str(file1),
        'file2': str(file2),
        'functions1': list(functions1.keys()),
        'functions2': list(functions2.keys()),
        'comparisons': {},
        'missing_in_file1': [],
        'missing_in_file2': [],
        'identical': [],
        'different': []
    }
    
    # Find missing functions
    all_functions = set(functions1.keys()) | set(functions2.keys())
    
    for func_name in all_functions:
        if func_name not in functions1:
            results['missing_in_file1'].append(func_name)
        elif func_name not in functions2:
            results['missing_in_file2'].append(func_name)
        else:
            # Compare the functions
            is_identical, differences = compare_functions(
                functions1[func_name], 
                functions2[func_name], 
                normalize_names
            )
            
            results['comparisons'][func_name] = {
                'identical': is_identical,
                'differences': differences
            }
            
            if is_identical:
                results['identical'].append(func_name)
            else:
                results['different'].append(func_name)
    
    return results


def print_comparison_report(results: Dict[str, Any]):
    """Print a formatted comparison report."""
    print("\n" + "="*80)
    print("AST COMPARISON REPORT")
    print("="*80)
    
    print(f"\nFile 1: {results['file1']}")
    print(f"File 2: {results['file2']}")
    
    print(f"\nFunctions in File 1: {len(results['functions1'])}")
    print(f"Functions in File 2: {len(results['functions2'])}")
    
    if results['missing_in_file1']:
        print(f"\n❌ Missing in File 1: {results['missing_in_file1']}")
    
    if results['missing_in_file2']:
        print(f"\n❌ Missing in File 2: {results['missing_in_file2']}")
    
    if results['identical']:
        print(f"\n✅ Identical Functions ({len(results['identical'])}): {results['identical']}")
    
    if results['different']:
        print(f"\n⚠️  Different Functions ({len(results['different'])}):")
        for func_name in results['different']:
            print(f"  - {func_name}:")
            for diff in results['comparisons'][func_name]['differences']:
                print(f"    • {diff}")
    
    # Summary
    total_compared = len(results['comparisons'])
    identical_count = len(results['identical'])
    if total_compared > 0:
        print(f"\n📊 Summary: {identical_count}/{total_compared} functions identical ({identical_count/total_compared*100:.1f}%)")
    
    print("="*80)


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) >= 3:
        file1 = Path(sys.argv[1])
        file2 = Path(sys.argv[2])
        normalize = len(sys.argv) > 3 and sys.argv[3].lower() == 'true'
        
        results = compare_modules(file1, file2, normalize)
        print_comparison_report(results)
    else:
        print("Usage: python ast_comparison.py <file1.py> <file2.py> [normalize_names]")
        print("Example: python ast_comparison.py src/working_calendar.py xlwings_lite/working_calendar_lite.py")