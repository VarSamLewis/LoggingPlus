"""
Tests for Logger wrapper class
"""

import logging
import pytest
from src.loggingplus import Logger, LoggerConfig


class TestLogger:
    """Test cases for Logger wrapper class."""

    @pytest.fixture
    def logger(self, tmp_path):
        """Create a test logger instance."""
        log_file = tmp_path / "test.log"
        config = LoggerConfig(name="test_logger", level=logging.DEBUG)
        config.add_file_handler(str(log_file))
        return Logger(config.get_logger()), log_file

    def test_init(self):
        """Test Logger initialization."""
        base_logger = logging.getLogger("test_init")
        logger = Logger(base_logger)
        assert logger._logger is base_logger

    def test_debug(self, logger):
        """Test debug logging method."""
        logger_obj, log_file = logger
        logger_obj.debug("debug message")

        content = log_file.read_text()
        assert "DEBUG" in content
        assert "debug message" in content

    def test_info(self, logger):
        """Test info logging method."""
        logger_obj, log_file = logger
        logger_obj.info("info message")

        content = log_file.read_text()
        assert "INFO" in content
        assert "info message" in content

    def test_warning(self, logger):
        """Test warning logging method."""
        logger_obj, log_file = logger
        logger_obj.warning("warning message")

        content = log_file.read_text()
        assert "WARNING" in content
        assert "warning message" in content

    def test_warn_alias(self, logger):
        """Test warn as alias for warning."""
        logger_obj, log_file = logger
        logger_obj.warn("warn message")

        content = log_file.read_text()
        assert "WARNING" in content
        assert "warn message" in content

    def test_error(self, logger):
        """Test error logging method."""
        logger_obj, log_file = logger
        logger_obj.error("error message")

        content = log_file.read_text()
        assert "ERROR" in content
        assert "error message" in content

    def test_critical(self, logger):
        """Test critical logging method."""
        logger_obj, log_file = logger
        logger_obj.critical("critical message")

        content = log_file.read_text()
        assert "CRITICAL" in content
        assert "critical message" in content

    def test_exception(self, logger):
        """Test exception logging with traceback."""
        logger_obj, log_file = logger

        try:
            raise ValueError("test error")
        except ValueError:
            logger_obj.exception("exception occurred")

        content = log_file.read_text()
        assert "ERROR" in content
        assert "exception occurred" in content
        assert "ValueError: test error" in content
        assert "Traceback" in content

    def test_log_with_custom_level(self, logger):
        """Test log method with custom level."""
        logger_obj, log_file = logger
        logger_obj.log(logging.WARNING, "custom level message")

        content = log_file.read_text()
        assert "WARNING" in content
        assert "custom level message" in content

    def test_set_level(self, logger):
        """Test changing logger level."""
        logger_obj, log_file = logger
        logger_obj.set_level(logging.WARNING)

        # Debug should not appear
        logger_obj.debug("should not appear")
        # Warning should appear
        logger_obj.warning("should appear")

        content = log_file.read_text()
        assert "should not appear" not in content
        assert "should appear" in content

    def test_level_property_getter(self, logger):
        """Test getting logger level via property."""
        logger_obj, log_file = logger
        assert logger_obj.level == logging.DEBUG

        logger_obj.set_level(logging.INFO)
        assert logger_obj.level == logging.INFO

    def test_formatted_logging(self, logger):
        """Test logging with format arguments."""
        logger_obj, log_file = logger
        logger_obj.info("User %s logged in", "john")

        content = log_file.read_text()
        assert "User john logged in" in content

    def test_extra_kwargs(self, logger):
        """Test logging with extra keyword arguments."""
        logger_obj, log_file = logger

        # Using extra to add custom fields
        logger_obj.info("message with extra", extra={"user_id": 123})

        # Should not raise an error
        content = log_file.read_text()
        assert "message with extra" in content

    def test_stack_info(self, logger):
        """Test logging with stack_info parameter."""
        logger_obj, log_file = logger
        logger_obj.info("message with stack", stack_info=True)

        content = log_file.read_text()
        assert "message with stack" in content
        assert "Stack (most recent call last)" in content

    def test_multiple_messages(self, logger):
        """Test logging multiple messages."""
        logger_obj, log_file = logger

        logger_obj.debug("message 1")
        logger_obj.info("message 2")
        logger_obj.warning("message 3")
        logger_obj.error("message 4")

        content = log_file.read_text()
        assert "message 1" in content
        assert "message 2" in content
        assert "message 3" in content
        assert "message 4" in content

    def test_unicode_messages(self, logger):
        """Test logging with unicode characters."""
        logger_obj, log_file = logger
        logger_obj.info("Unicode: \u2713 \u2717 \u00e9 \u4e2d\u6587")

        content = log_file.read_text(encoding='utf-8')
        assert "\u2713" in content
        assert "\u4e2d\u6587" in content

    def test_multiline_messages(self, logger):
        """Test logging multiline messages."""
        logger_obj, log_file = logger
        multiline = "Line 1\nLine 2\nLine 3"
        logger_obj.info(multiline)

        content = log_file.read_text()
        assert "Line 1" in content
        assert "Line 2" in content
        assert "Line 3" in content
