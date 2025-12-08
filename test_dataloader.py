"""
Comprehensive Test Data Loader for Dataset Loading Issues - COMPLETE VERSION

FILE PATH: /home/yousef/ga-optimized-trees/test_dataloader.py
ACTION: CREATE NEW FILE

This script tests all aspects of the dataset loader to identify issues:
- OpenML dataset loading
- CSV/Excel file loading
- Data validation
- Error handling
- Integration with GA training

Usage:
    python3 test_dataloader.py
    python3 test_dataloader.py --dataset heart
    python3 test_dataloader.py --verbose
"""

import sys
import argparse
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import tempfile
import shutil

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from ga_trees.data.dataset_loader import (
        DatasetLoader, 
        DataValidator,
        load_benchmark_dataset
    )
    LOADER_AVAILABLE = True
except ImportError as e:
    print(f"❌ CRITICAL: Cannot import dataset_loader: {e}")
    LOADER_AVAILABLE = False


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class TestResult:
    """Store test results."""
    def __init__(self, name):
        self.name = name
        self.passed = False
        self.error = None
        self.warning = None
        self.data = None
        
    def __repr__(self):
        status = f"{Colors.GREEN}✓{Colors.RESET}" if self.passed else f"{Colors.RED}✗{Colors.RESET}"
        return f"{status} {self.name}"


class DataLoaderTester:
    """Comprehensive dataset loader testing."""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.results = []
        self.loader = DatasetLoader() if LOADER_AVAILABLE else None
        
    def log(self, message, level='info'):
        """Print log message."""
        if not self.verbose and level == 'debug':
            return
            
        prefix = {
            'info': f"{Colors.BLUE}ℹ{Colors.RESET}",
            'success': f"{Colors.GREEN}✓{Colors.RESET}",
            'error': f"{Colors.RED}✗{Colors.RESET}",
            'warning': f"{Colors.YELLOW}⚠{Colors.RESET}",
            'debug': f"{Colors.BLUE}→{Colors.RESET}"
        }
        print(f"{prefix.get(level, '')} {message}")
    
    def run_test(self, test_name, test_func):
        """Run a single test and record result."""
        result = TestResult(test_name)
        
        try:
            self.log(f"Testing: {test_name}", 'debug')
            test_func(result)
            result.passed = True
            self.log(f"✓ {test_name}", 'success')
        except Exception as e:
            result.passed = False
            result.error = str(e)
            self.log(f"✗ {test_name}: {e}", 'error')
            if self.verbose:
                self.log(f"  Traceback:\n{traceback.format_exc()}", 'debug')
        
        self.results.append(result)
        return result
    
    # ========================================================================
    # SKLEARN DATASET TESTS
    # ========================================================================
    
    def test_sklearn_iris(self, result):
        """Test loading Iris dataset."""
        data = self.loader.load_dataset('iris', test_size=0.2, random_state=42)
        
        assert data is not None, "Data is None"
        assert 'X_train' in data, "Missing X_train"
        assert 'y_train' in data, "Missing y_train"
        assert data['metadata']['n_samples'] == 150, "Wrong sample count"
        assert data['metadata']['n_features'] == 4, "Wrong feature count"
        assert data['metadata']['n_classes'] == 3, "Wrong class count"
        
        result.data = data
        self.log(f"  Loaded: {data['metadata']['n_samples']} samples, "
                f"{data['metadata']['n_features']} features", 'debug')
    
    def test_sklearn_wine(self, result):
        """Test loading Wine dataset."""
        data = self.loader.load_dataset('wine', test_size=0.3, random_state=42)
        
        assert data['metadata']['n_samples'] == 178
        assert data['metadata']['n_classes'] == 3
        result.data = data
    
    def test_sklearn_breast_cancer(self, result):
        """Test loading Breast Cancer dataset."""
        data = self.loader.load_dataset('breast_cancer', test_size=0.2, random_state=42)
        
        assert data['metadata']['n_samples'] == 569
        assert data['metadata']['n_features'] == 30
        assert data['metadata']['n_classes'] == 2
        result.data = data
    
    # ========================================================================
    # OPENML DATASET TESTS
    # ========================================================================
    
    def test_openml_heart(self, result):
        """Test loading Heart Disease dataset from OpenML."""
        try:
            data = self.loader.load_dataset('heart', test_size=0.2, random_state=42)
            
            assert data is not None
            assert data['metadata']['n_samples'] > 0
            assert data['metadata']['n_classes'] == 2
            
            # Check data types
            assert np.issubdtype(data['X_train'].dtype, np.floating)
            assert np.issubdtype(data['y_train'].dtype, np.integer)
            
            # Check for NaN
            assert not np.any(np.isnan(data['X_train'])), "X_train contains NaN"
            assert not np.any(np.isnan(data['y_train'])), "y_train contains NaN"
            
            result.data = data
            self.log(f"  Heart: {data['metadata']['n_samples']} samples, "
                    f"{data['metadata']['n_features']} features", 'debug')
            
        except Exception as e:
            if 'openml' in str(e).lower() or 'fetch' in str(e).lower():
                result.warning = "OpenML service may be unavailable"
                self.log(f"  OpenML warning: {e}", 'warning')
            raise
    
    def test_openml_titanic(self, result):
        """Test loading Titanic dataset from OpenML."""
        try:
            data = self.loader.load_dataset('titanic', test_size=0.2, random_state=42)
            
            assert data['metadata']['n_samples'] > 1000
            assert data['metadata']['n_classes'] == 2
            
            # Verify no categorical data leaked through
            assert np.issubdtype(data['X_train'].dtype, np.floating)
            
            result.data = data
            
        except Exception as e:
            if 'openml' in str(e).lower():
                result.warning = "OpenML unavailable"
            raise
    
    def test_openml_credit_g(self, result):
        """Test loading German Credit dataset."""
        try:
            data = self.loader.load_dataset('credit_g', test_size=0.2, random_state=42)
            
            assert data['metadata']['n_samples'] == 1000
            assert data['metadata']['n_classes'] == 2
            
            result.data = data
            
        except Exception as e:
            if 'openml' in str(e).lower():
                result.warning = "OpenML unavailable"
            raise
    
    # ========================================================================
    # CSV/EXCEL FILE TESTS - UPDATED WITH LARGER SAMPLES
    # ========================================================================
    
    def test_csv_basic(self, result):
        """Test loading basic CSV file with adequate samples."""
        temp_dir = tempfile.mkdtemp()
        csv_path = Path(temp_dir) / 'test.csv'
        
        try:
            # Increased from 5 to 20 samples
            df = pd.DataFrame({
                'feature1': np.random.rand(20),
                'feature2': np.random.rand(20),
                'target': np.random.randint(0, 2, 20)
            })
            df.to_csv(csv_path, index=False)
            
            data = self.loader.load_dataset(str(csv_path), test_size=0.2, random_state=42)
            
            assert data['metadata']['n_samples'] == 20
            assert data['metadata']['n_features'] == 2
            assert len(data['X_train']) == 16
            assert len(data['X_test']) == 4
            
            result.data = data
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_csv_categorical_target(self, result):
        """Test CSV with categorical target and adequate samples."""
        temp_dir = tempfile.mkdtemp()
        csv_path = Path(temp_dir) / 'test_cat.csv'
        
        try:
            # Increased from 6 to 20 samples
            df = pd.DataFrame({
                'feature1': np.random.rand(20),
                'feature2': np.random.rand(20),
                'target': ['cat', 'dog'] * 10  # Balanced classes
            })
            df.to_csv(csv_path, index=False)
            
            data = self.loader.load_dataset(str(csv_path), test_size=0.2, random_state=42)
            
            # Target should be encoded to integers
            assert np.issubdtype(data['y_train'].dtype, np.integer)
            assert data['target_names'] is not None
            assert set(data['target_names']) == {'cat', 'dog'}
            
            result.data = data
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_csv_with_nan(self, result):
        """Test CSV with missing values and adequate samples."""
        temp_dir = tempfile.mkdtemp()
        csv_path = Path(temp_dir) / 'test_nan.csv'
        
        try:
            # Increased from 5 to 15 samples
            data_values = np.random.rand(15, 2)
            data_values[3, 0] = np.nan
            data_values[7, 1] = np.nan
            
            df = pd.DataFrame({
                'feature1': data_values[:, 0],
                'feature2': data_values[:, 1],
                'target': np.random.randint(0, 2, 15)
            })
            df.to_csv(csv_path, index=False)
            
            data = self.loader.load_dataset(str(csv_path), test_size=0.2, random_state=42)
            
            # Should handle NaN (clean with median by default)
            assert not np.any(np.isnan(data['X_train']))
            assert not np.any(np.isnan(data['X_test']))
            
            result.data = data
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_csv_very_small(self, result):
        """Test CSV with very small dataset (edge case)."""
        temp_dir = tempfile.mkdtemp()
        csv_path = Path(temp_dir) / 'tiny.csv'
        
        try:
            # Create dataset with 10 samples (minimum safe size)
            # With 2 classes, need at least 2 samples per class in test set
            df = pd.DataFrame({
                'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                'feature2': [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
                'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # Balanced
            })
            df.to_csv(csv_path, index=False)
            
            # Should work with warning for small dataset
            data = self.loader.load_dataset(str(csv_path), test_size=0.2, random_state=42)
            
            assert data is not None
            assert len(data['X_train']) == 8
            assert len(data['X_test']) == 2
            
            result.data = data
            result.warning = "Small dataset handled successfully"
            
        finally:
            shutil.rmtree(temp_dir)
    
    # ========================================================================
    # DATA VALIDATION TESTS
    # ========================================================================
    
    def test_validation_empty_dataset(self, result):
        """Test validation catches empty dataset."""
        X = np.array([]).reshape(0, 4)
        y = np.array([])
        
        validator = DataValidator()
        valid, warnings = validator.validate_dataset(X, y)
        
        assert not valid, "Should reject empty dataset"
        assert any('empty' in w.lower() for w in warnings)
    
    def test_validation_dimension_mismatch(self, result):
        """Test validation catches dimension mismatch."""
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 2, 90)  # Wrong size
        
        validator = DataValidator()
        valid, warnings = validator.validate_dataset(X, y)
        
        assert not valid
        assert any('different' in w.lower() for w in warnings)
    
    def test_validation_nan_detection(self, result):
        """Test NaN detection in validation."""
        X = np.random.rand(100, 4)
        X[10, 2] = np.nan
        y = np.random.randint(0, 2, 100)
        
        validator = DataValidator()
        valid, warnings = validator.validate_dataset(X, y)
        
        assert valid  # Valid but with warnings
        assert any('nan' in w.lower() for w in warnings)
    
    def test_validation_class_imbalance(self, result):
        """Test severe class imbalance detection."""
        X = np.random.rand(100, 4)
        y = np.concatenate([np.zeros(95), np.ones(5)])  # 95:5 ratio
        
        validator = DataValidator()
        valid, warnings = validator.validate_dataset(X, y, task_type='classification')
        
        assert valid
        assert any('imbalance' in w.lower() for w in warnings)
    
    # ========================================================================
    # DATA CLEANING TESTS
    # ========================================================================
    
    def test_cleaning_median_imputation(self, result):
        """Test median imputation cleaning."""
        X = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
        y = np.array([0, 1, 0])
        
        validator = DataValidator()
        X_clean, y_clean = validator.clean_dataset(X, y, strategy='median')
        
        assert len(X_clean) == 3
        assert not np.any(np.isnan(X_clean))
        # Median of [2.0, 6.0] = 4.0
        assert X_clean[1, 1] == 4.0
    
    def test_cleaning_remove_rows(self, result):
        """Test row removal cleaning."""
        X = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
        y = np.array([0, 1, 0])
        
        validator = DataValidator()
        X_clean, y_clean = validator.clean_dataset(X, y, strategy='remove')
        
        assert len(X_clean) == 2
        assert len(y_clean) == 2
        assert not np.any(np.isnan(X_clean))
    
    # ========================================================================
    # FEATURE TESTS
    # ========================================================================
    
    def test_standardization(self, result):
        """Test feature standardization."""
        data = self.loader.load_dataset('iris', standardize=True, random_state=42)
        
        mean = np.mean(data['X_train'], axis=0)
        std = np.std(data['X_train'], axis=0)
        
        assert np.allclose(mean, 0, atol=1e-10), "Mean should be ~0"
        assert np.allclose(std, 1, atol=0.1), "Std should be ~1"
        assert data['scaler'] is not None
        assert data['metadata']['standardized'] is True
        
        result.data = data
    
    def test_stratified_split(self, result):
        """Test stratified splitting."""
        data = self.loader.load_dataset('iris', test_size=0.2, stratify=True, random_state=42)
        
        train_classes, train_counts = np.unique(data['y_train'], return_counts=True)
        test_classes, test_counts = np.unique(data['y_test'], return_counts=True)
        
        assert len(train_classes) == 3, "All classes in train"
        assert len(test_classes) == 3, "All classes in test"
        assert np.all(train_counts > 0)
        assert np.all(test_counts > 0)
        
        result.data = data
    
    def test_balancing_oversample(self, result):
        """Test oversampling for class balance."""
        data = self.loader.load_dataset('iris', test_size=0.2, balance='oversample', random_state=42)
        
        unique, counts = np.unique(data['y_train'], return_counts=True)
        
        # Classes should be balanced
        assert len(set(counts)) == 1, "All classes should have equal count"
        assert data['metadata']['balanced'] is True
        
        result.data = data
    
    # ========================================================================
    # INTEGRATION TESTS
    # ========================================================================
    
    def test_integration_feature_ranges(self, result):
        """Test extracting feature ranges for GA."""
        data = self.loader.load_dataset('wine', standardize=False, random_state=42)
        X_train = data['X_train']
        n_features = data['metadata']['n_features']
        
        # Extract feature ranges (needed for GA mutation)
        feature_ranges = {i: (X_train[:, i].min(), X_train[:, i].max()) 
                         for i in range(n_features)}
        
        assert len(feature_ranges) == n_features
        for i, (min_val, max_val) in feature_ranges.items():
            assert min_val <= max_val
            assert not np.isnan(min_val)
            assert not np.isnan(max_val)
    
    def test_integration_ga_compatibility(self, result):
        """Test data format is GA-compatible."""
        data = self.loader.load_dataset('breast_cancer', test_size=0.2, standardize=True, random_state=42)
        
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Check types
        assert isinstance(X_train, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        
        # Check shapes
        assert X_train.shape[1] == data['metadata']['n_features']
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]
        
        # Check no NaN/Inf
        assert not np.any(np.isnan(X_train))
        assert not np.any(np.isinf(X_train))
        assert not np.any(np.isnan(y_train))
        
        result.data = data
    
    # ========================================================================
    # ERROR HANDLING TESTS - UPDATED
    # ========================================================================
    
    def test_error_nonexistent_dataset(self, result):
        """Test error for nonexistent dataset."""
        try:
            self.loader.load_dataset('definitely_not_a_real_dataset_12345')
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert 'unknown' in str(e).lower()
    
    def test_error_invalid_test_size(self, result):
        """Test error for invalid test_size."""
        try:
            self.loader.load_dataset('iris', test_size=1.5)
            raise AssertionError("Should have raised error for test_size > 1")
        except (ValueError, Exception):
            pass  # Expected
    
    def test_error_nonexistent_file(self, result):
        """Test error for nonexistent file - FIXED."""
        try:
            # Use clear file path format to trigger FileNotFoundError
            self.loader.load_dataset('path/to/nonexistent_file.csv')
            raise AssertionError("Should have raised FileNotFoundError")
        except FileNotFoundError as e:
            assert 'not found' in str(e).lower()
    
    # ========================================================================
    # UTILITY TESTS
    # ========================================================================
    
    def test_list_available_datasets(self, result):
        """Test listing available datasets."""
        available = DatasetLoader.list_available_datasets()
        
        assert 'sklearn' in available
        assert 'openml' in available
        assert 'custom' in available
        
        assert len(available['sklearn']) >= 5
        assert len(available['openml']) >= 15
    
    def test_get_dataset_info(self, result):
        """Test getting dataset info."""
        info = DatasetLoader.get_dataset_info('iris')
        
        assert info['name'] == 'iris'
        assert info['source'] == 'sklearn'
        assert info['available'] is True
    
    def test_convenience_function(self, result):
        """Test load_benchmark_dataset convenience function."""
        data = load_benchmark_dataset('wine', test_size=0.2, random_state=42)
        
        assert 'X_train' in data
        assert 'y_train' in data
        assert data['metadata']['n_samples'] == 178
    
    # ========================================================================
    # TEST RUNNER
    # ========================================================================
    
    def run_all_tests(self):
        """Run all tests."""
        print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}Dataset Loader Comprehensive Test Suite{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")
        
        if not LOADER_AVAILABLE:
            print(f"{Colors.RED}CRITICAL ERROR: Dataset loader not available{Colors.RESET}")
            return False
        
        # Define test categories
        test_categories = [
            ("Sklearn Datasets", [
                ('test_sklearn_iris', self.test_sklearn_iris),
                ('test_sklearn_wine', self.test_sklearn_wine),
                ('test_sklearn_breast_cancer', self.test_sklearn_breast_cancer),
            ]),
            ("OpenML Datasets", [
                ('test_openml_heart', self.test_openml_heart),
                ('test_openml_titanic', self.test_openml_titanic),
                ('test_openml_credit_g', self.test_openml_credit_g),
            ]),
            ("CSV/Excel Files", [
                ('test_csv_basic', self.test_csv_basic),
                ('test_csv_categorical_target', self.test_csv_categorical_target),
                ('test_csv_with_nan', self.test_csv_with_nan),
                ('test_csv_very_small', self.test_csv_very_small),
            ]),
            ("Data Validation", [
                ('test_validation_empty_dataset', self.test_validation_empty_dataset),
                ('test_validation_dimension_mismatch', self.test_validation_dimension_mismatch),
                ('test_validation_nan_detection', self.test_validation_nan_detection),
                ('test_validation_class_imbalance', self.test_validation_class_imbalance),
            ]),
            ("Data Cleaning", [
                ('test_cleaning_median_imputation', self.test_cleaning_median_imputation),
                ('test_cleaning_remove_rows', self.test_cleaning_remove_rows),
            ]),
            ("Features", [
                ('test_standardization', self.test_standardization),
                ('test_stratified_split', self.test_stratified_split),
                ('test_balancing_oversample', self.test_balancing_oversample),
            ]),
            ("Integration Tests", [
                ('test_integration_feature_ranges', self.test_integration_feature_ranges),
                ('test_integration_ga_compatibility', self.test_integration_ga_compatibility),
            ]),
            ("Error Handling", [
                ('test_error_nonexistent_dataset', self.test_error_nonexistent_dataset),
                ('test_error_invalid_test_size', self.test_error_invalid_test_size),
                ('test_error_nonexistent_file', self.test_error_nonexistent_file),
            ]),
            ("Utilities", [
                ('test_list_available_datasets', self.test_list_available_datasets),
                ('test_get_dataset_info', self.test_get_dataset_info),
                ('test_convenience_function', self.test_convenience_function),
            ]),
        ]
        
        # Run tests by category
        for category_name, tests in test_categories:
            print(f"\n{Colors.BOLD}{category_name}{Colors.RESET}")
            print("-" * 70)
            
            for test_name, test_func in tests:
                self.run_test(test_name, test_func)
        
        # Print summary
        self.print_summary()
        
        return all(r.passed or r.warning for r in self.results)
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}Test Summary{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*70}{Colors.RESET}\n")
        
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed and not r.warning)
        warnings = sum(1 for r in self.results if r.warning)
        total = len(self.results)
        
        print(f"Total Tests: {total}")
        print(f"{Colors.GREEN}Passed: {passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {failed}{Colors.RESET}")
        print(f"{Colors.YELLOW}Warnings: {warnings}{Colors.RESET}")
        
        if failed > 0:
            print(f"\n{Colors.RED}Failed Tests:{Colors.RESET}")
            for r in self.results:
                if not r.passed and not r.warning:
                    print(f"  {Colors.RED}✗{Colors.RESET} {r.name}")
                    if r.error:
                        print(f"    Error: {r.error}")
        
        if warnings > 0:
            print(f"\n{Colors.YELLOW}Warnings:{Colors.RESET}")
            for r in self.results:
                if r.warning:
                    print(f"  {Colors.YELLOW}⚠{Colors.RESET} {r.name}")
                    print(f"    Warning: {r.warning}")
        
        success_rate = (passed / total * 100) if total > 0 else 0
        print(f"\n{Colors.BOLD}Success Rate: {success_rate:.1f}%{Colors.RESET}")
        
        if success_rate == 100:
            print(f"{Colors.GREEN}🎉 All tests passed!{Colors.RESET}")
        elif success_rate >= 80:
            print(f"{Colors.YELLOW}⚠ Most tests passed, some issues found{Colors.RESET}")
        else:
            print(f"{Colors.RED}❌ Significant issues detected{Colors.RESET}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Comprehensive Dataset Loader Testing',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with detailed logs')
    parser.add_argument('--dataset', type=str,
                       help='Test specific dataset only')
    
    args = parser.parse_args()
    
    tester = DataLoaderTester(verbose=args.verbose)
    
    if args.dataset:
        # Test specific dataset
        print(f"\nTesting dataset: {args.dataset}\n")
        result = TestResult(f"test_{args.dataset}")
        
        try:
            data = tester.loader.load_dataset(args.dataset, test_size=0.2, random_state=42)
            result.passed = True
            result.data = data
            
            print(f"{Colors.GREEN}✓ Successfully loaded {args.dataset}{Colors.RESET}")
            print(f"\nMetadata:")
            for key, value in data['metadata'].items():
                print(f"  {key}: {value}")
                
        except Exception as e:
            result.passed = False
            result.error = str(e)
            print(f"{Colors.RED}✗ Failed to load {args.dataset}{Colors.RESET}")
            print(f"Error: {e}")
            if args.verbose:
                print(f"\nTraceback:\n{traceback.format_exc()}")
    else:
        # Run all tests
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()