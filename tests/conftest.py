"""
Pytest configuration and shared fixtures
"""

import logging
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def reset_logging():
    """
    Reset logging configuration before each test to prevent interference.
    This fixture runs automatically for all tests.
    """
    # Clear all loggers before test
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        # Close handlers properly before clearing
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)
        logger.propagate = True

    yield

    # Clean up after test
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        # Close handlers properly before clearing
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        logger.setLevel(logging.NOTSET)


@pytest.fixture
def temp_log_dir(tmp_path):
    """
    Create a temporary directory for log files.
    Returns a Path object to the temporary directory.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir


@pytest.fixture
def sample_log_data():
    """
    Provide sample log data for testing.
    """
    return {
        "messages": [
            "This is a debug message",
            "This is an info message",
            "This is a warning message",
            "This is an error message",
            "This is a critical message",
        ],
        "structured": {
            "user_id": 123,
            "action": "login",
            "status": "success",
            "ip": "192.168.1.1",
        },
    }


@pytest.fixture
def sample_csv_data():
    """
    Provide sample CSV data for testing.
    """
    return [
        {"name": "Alice", "age": 30, "city": "New York"},
        {"name": "Bob", "age": 25, "city": "Los Angeles"},
        {"name": "Charlie", "age": 35, "city": "Chicago"},
    ]


@pytest.fixture
def sample_json_data():
    """
    Provide sample JSON data for testing.
    """
    return {
        "users": [
            {"id": 1, "name": "Alice", "active": True},
            {"id": 2, "name": "Bob", "active": False},
        ],
        "metadata": {"version": "1.0", "timestamp": "2025-01-01T00:00:00"},
    }


@pytest.fixture
def cleanup_log_files():
    """
    Fixture to track and clean up log files created during tests.
    """
    files_to_cleanup = []

    def register_file(filepath):
        """Register a file for cleanup."""
        files_to_cleanup.append(Path(filepath))

    yield register_file

    # Cleanup after test
    for file_path in files_to_cleanup:
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception:
                pass  # Ignore cleanup errors


@pytest.fixture
def mock_logger_output(tmp_path):
    """
    Create a temporary log file and return both the logger and file path
    for easy output verification.
    """
    from src.loggingplus import LoggerConfig, Logger

    log_file = tmp_path / "test_output.log"
    config = LoggerConfig(name="mock_logger", level=logging.DEBUG)
    config.add_file_handler(str(log_file))
    logger = Logger(config.get_logger())

    return logger, log_file


def pytest_configure(config):
    """
    Pytest configuration hook - runs once before all tests.
    """
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers automatically based on test names.
    """
    for item in items:
        # Add 'slow' marker to tests with 'slow' in their name
        if "slow" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)

        # Add 'integration' marker to tests with 'integration' in their name
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
