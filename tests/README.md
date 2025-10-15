# LoggingPlus Tests

This directory contains comprehensive tests for the LoggingPlus library.

## Test Structure

- `test_logger_config.py` - Tests for LoggerConfig class
- `test_logger.py` - Tests for Logger wrapper class
- `test_logger_persistence.py` - Tests for LoggerPersistence class
- `test_setup_logger.py` - Tests for setup_logger convenience function
- `test_integration.py` - Integration tests combining multiple components
- `conftest.py` - Shared pytest fixtures and configuration

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_logger_config.py
```

### Run specific test class
```bash
pytest tests/test_logger_config.py::TestLoggerConfig
```

### Run specific test
```bash
pytest tests/test_logger_config.py::TestLoggerConfig::test_init_default_params
```

### Run with verbose output
```bash
pytest -v
```

### Run with coverage
```bash
pytest --cov=src/loggingplus --cov-report=html
```

### Run only fast tests (exclude slow tests)
```bash
pytest -m "not slow"
```

### Run only unit tests (exclude integration tests)
```bash
pytest -m "not integration"
```

## Test Markers

- `@pytest.mark.slow` - Marks tests as slow-running
- `@pytest.mark.integration` - Marks tests as integration tests

## Fixtures

Common fixtures available in `conftest.py`:

- `tmp_path` - Pytest built-in fixture for temporary directory
- `temp_log_dir` - Temporary directory specifically for log files
- `sample_log_data` - Sample log messages and structured data
- `sample_csv_data` - Sample CSV data for testing
- `sample_json_data` - Sample JSON data for testing
- `mock_logger_output` - Pre-configured logger with file output for testing
- `reset_logging` - Auto-use fixture that resets logging state between tests

## Writing New Tests

When writing new tests:

1. Use descriptive test names starting with `test_`
2. Use fixtures from `conftest.py` where applicable
3. Use `tmp_path` fixture for file operations
4. Add docstrings explaining what the test verifies
5. Group related tests in classes
6. Add appropriate markers for slow or integration tests

Example:
```python
def test_my_feature(tmp_path, sample_json_data):
    """Test that my feature works correctly."""
    # Arrange
    log_file = tmp_path / "test.log"

    # Act
    # ... test code ...

    # Assert
    assert result == expected
```

## Test Coverage

To generate a coverage report:

```bash
pytest --cov=src/loggingplus --cov-report=html
```

Then open `htmlcov/index.html` in your browser.

## Continuous Integration

These tests are designed to run in CI/CD environments. Ensure all tests pass before merging changes.
