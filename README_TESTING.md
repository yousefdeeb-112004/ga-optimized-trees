# 🧪 Testing Guide - Dataset Loader with Pytest

## 📁 File Locations

### Required Files:

```
/home/yousef/ga-optimized-trees/
│
├── tests/
│   └── unit/
│       ├── __init__.py
│       └── test_dataset_loader.py          ← NEW/REPLACE THIS
│
├── pytest.ini                               ← CREATE THIS
└── ...
```

---

## 🚀 Quick Start

### Install pytest (if not already installed):
```bash
pip install pytest pytest-cov
```

### Run all tests:
```bash
cd /home/yousef/ga-optimized-trees
pytest tests/unit/test_dataset_loader.py -v
```

---

## 📊 Common Testing Commands

### 1. Run all tests with verbose output:
```bash
pytest tests/unit/test_dataset_loader.py -v
```

### 2. Run with coverage report:
```bash
pytest tests/unit/test_dataset_loader.py -v --cov=src/ga_trees/data --cov-report=html
```

### 3. Run only fast tests (skip slow OpenML tests):
```bash
pytest tests/unit/test_dataset_loader.py -v -m "not slow"
```

### 4. Run only slow tests:
```bash
pytest tests/unit/test_dataset_loader.py -v -m "slow"
```

### 5. Run specific test class:
```bash
pytest tests/unit/test_dataset_loader.py::TestSklearnDatasets -v
```

### 6. Run specific test:
```bash
pytest tests/unit/test_dataset_loader.py::TestSklearnDatasets::test_load_iris -v
```

### 7. Run tests matching a pattern:
```bash
pytest tests/unit/test_dataset_loader.py -v -k "csv"
```

### 8. Stop at first failure:
```bash
pytest tests/unit/test_dataset_loader.py -v -x
```

### 9. Show local variables on failure:
```bash
pytest tests/unit/test_dataset_loader.py -v -l
```

### 10. Run with detailed output on failures:
```bash
pytest tests/unit/test_dataset_loader.py -v --tb=long
```

---

## 📦 Test Organization

### Test Classes:

| Class | Description | # Tests |
|-------|-------------|---------|
| `TestSklearnDatasets` | Sklearn built-in datasets | 3 |
| `TestOpenMLDatasets` | OpenML datasets (marked as slow) | 3 |
| `TestCSVLoading` | CSV/Excel file loading | 5 |
| `TestDataValidation` | Data validation checks | 7 |
| `TestDataCleaning` | Data cleaning strategies | 4 |
| `TestFeatures` | Feature processing | 3 |
| `TestIntegration` | GA integration tests | 2 |
| `TestErrorHandling` | Error handling | 5 |
| `TestUtilities` | Utility functions | 4 |
| `TestReproducibility` | Random seed tests | 2 |

**Total:** ~38 tests

---

## 🎯 Test Markers

Tests can be marked with categories:

### Available Markers:
- `@pytest.mark.slow` - Slow tests (e.g., OpenML downloads)
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.unit` - Unit tests

### Usage:
```bash
# Run all except slow tests (recommended for quick checks)
pytest tests/unit/test_dataset_loader.py -v -m "not slow"

# Run only slow tests
pytest tests/unit/test_dataset_loader.py -v -m "slow"
```

---

## 📈 Coverage Reports

### Generate HTML coverage report:
```bash
pytest tests/unit/test_dataset_loader.py -v --cov=src/ga_trees/data --cov-report=html

# View report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Generate terminal coverage report:
```bash
pytest tests/unit/test_dataset_loader.py -v --cov=src/ga_trees/data --cov-report=term-missing
```

### Example output:
```
---------- coverage: platform linux, python 3.10 -----------
Name                                   Stmts   Miss  Cover   Missing
--------------------------------------------------------------------
src/ga_trees/data/dataset_loader.py     450     12    97%   123-125, 456-460
--------------------------------------------------------------------
TOTAL                                    450     12    97%
```

---

## 🔧 Pytest Configuration

The `pytest.ini` file configures pytest behavior:

```ini
[pytest]
# Test discovery
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Test paths
testpaths = tests

# Markers
markers =
    slow: marks tests as slow
    integration: integration tests
    unit: unit tests
```

---

## 🎨 Example Test Output

### Successful run:
```
$ pytest tests/unit/test_dataset_loader.py -v

tests/unit/test_dataset_loader.py::TestSklearnDatasets::test_load_iris PASSED
tests/unit/test_dataset_loader.py::TestSklearnDatasets::test_load_wine PASSED
tests/unit/test_dataset_loader.py::TestSklearnDatasets::test_load_breast_cancer PASSED
tests/unit/test_dataset_loader.py::TestOpenMLDatasets::test_load_heart PASSED
...

========================= 38 passed in 45.2s =========================
```

### With coverage:
```
$ pytest tests/unit/test_dataset_loader.py -v --cov=src/ga_trees/data

========================= 38 passed in 45.2s =========================

---------- coverage: platform linux, python 3.10 -----------
Name                                   Stmts   Miss  Cover
----------------------------------------------------------
src/ga_trees/data/dataset_loader.py     450     12    97%
----------------------------------------------------------
TOTAL                                    450     12    97%
```

---

## 🐛 Debugging Failed Tests

### 1. Run with detailed traceback:
```bash
pytest tests/unit/test_dataset_loader.py -v --tb=long
```

### 2. Run with local variables:
```bash
pytest tests/unit/test_dataset_loader.py -v -l
```

### 3. Run with pdb (Python debugger):
```bash
pytest tests/unit/test_dataset_loader.py -v --pdb
```

### 4. Run specific failing test:
```bash
pytest tests/unit/test_dataset_loader.py::TestCSVLoading::test_load_csv_with_nan -v
```

---

## 🔄 Continuous Integration

### Add to `.github/workflows/ci.yml`:

```yaml
- name: Run dataset loader tests
  run: |
    pip install pytest pytest-cov
    pytest tests/unit/test_dataset_loader.py -v --cov=src/ga_trees/data --cov-report=xml
    
- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
```

---

## 📝 Writing New Tests

### Basic test template:
```python
def test_my_new_feature(loader):
    """Test description."""
    # Arrange
    data = loader.load_dataset('iris', test_size=0.2)
    
    # Act
    result = some_operation(data)
    
    # Assert
    assert result is not None
    assert result == expected_value
```

### Using fixtures:
```python
def test_with_csv_file(loader, temp_dir):
    """Test with temporary CSV file."""
    # temp_dir is automatically created and cleaned up
    csv_path = Path(temp_dir) / 'test.csv'
    # ... create CSV and test
```

---

## ⚡ Quick Reference

| Command | Description |
|---------|-------------|
| `pytest tests/unit/test_dataset_loader.py -v` | Run all tests |
| `pytest ... -v -m "not slow"` | Skip slow tests |
| `pytest ... -v -k "csv"` | Run tests matching "csv" |
| `pytest ... -v --cov=src/ga_trees/data` | With coverage |
| `pytest ... -v -x` | Stop at first failure |
| `pytest ... -v --tb=short` | Short traceback |
| `pytest ... -v --pdb` | Drop into debugger |

---

## 🎯 Expected Results

After running all tests, you should see:

```
========================= 38 passed in 45.2s =========================

Coverage: 97%
```

All tests should pass with high coverage (>95%).

---

## 🚀 Integration with Existing Tests

### Run all project tests:
```bash
# Run all unit tests
pytest tests/unit/ -v

# Run all tests (unit + integration)
pytest tests/ -v

# Run with coverage for entire project
pytest tests/ -v --cov=src/ga_trees --cov-report=html
```

---

## 📞 Troubleshooting

### Issue: ModuleNotFoundError
**Solution:**
```bash
# Make sure package is installed in editable mode
cd /home/yousef/ga-optimized-trees
pip install -e .
```

### Issue: Tests not discovered
**Solution:**
```bash
# Check file naming
# Files must be named: test_*.py
# Classes must be named: Test*
# Functions must be named: test_*

# Verify pytest can find tests
pytest --collect-only tests/unit/test_dataset_loader.py
```

### Issue: OpenML tests fail
**Solution:**
```bash
# Skip slow tests during development
pytest tests/unit/test_dataset_loader.py -v -m "not slow"
```

---

## 🎉 Summary

You now have a professional pytest test suite with:

- ✅ 38 comprehensive tests
- ✅ Organized test classes
- ✅ Fixtures for reusable test data
- ✅ Test markers (slow, integration, unit)
- ✅ Coverage reporting
- ✅ Easy debugging options
- ✅ CI/CD ready

Run tests with: `pytest tests/unit/test_dataset_loader.py -v`