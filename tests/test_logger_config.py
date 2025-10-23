"""
Tests for LoggerConfig class
"""

import logging
import sys
from pathlib import Path
import pytest
from src.loggingplus import LoggerConfig


class TestLoggerConfig:
    """Test cases for LoggerConfig class."""

    def test_init_default_params(self):
        """Test LoggerConfig initialization with default parameters."""
        config = LoggerConfig()
        # Default name uses the module's __name__ which is 'src.loggingplus.core'
        assert config.name == 'src.loggingplus.core'
        assert config.level == logging.INFO
        assert "%(asctime)s" in config.format_string
        assert config.logger is not None
        assert isinstance(config.formatter, logging.Formatter)

    def test_init_custom_params(self):
        """Test LoggerConfig initialization with custom parameters."""
        custom_format = "%(levelname)s - %(message)s"
        config = LoggerConfig(
            name="test_logger", level=logging.DEBUG, format_string=custom_format
        )
        assert config.name == "test_logger"
        assert config.level == logging.DEBUG
        assert config.format_string == custom_format

    def test_init_clears_existing_handlers(self):
        """Test that initialization clears existing handlers."""
        # Create logger with handler
        logger = logging.getLogger("test_clear_handlers")
        logger.addHandler(logging.StreamHandler())
        assert len(logger.handlers) > 0

        # Create config with same logger name
        config = LoggerConfig(name="test_clear_handlers")
        assert len(config.logger.handlers) == 0

    def test_add_console_handler(self, tmp_path):
        """Test adding console handler."""
        config = LoggerConfig(name="test_console")
        result = config.add_console_handler()

        assert result is config  # Method chaining
        assert len(config.logger.handlers) == 1
        assert isinstance(config.logger.handlers[0], logging.StreamHandler)

    def test_add_console_handler_custom_level(self):
        """Test console handler with custom level."""
        config = LoggerConfig(name="test_console_level", level=logging.INFO)
        config.add_console_handler(level=logging.ERROR)

        handler = config.logger.handlers[0]
        assert handler.level == logging.ERROR

    def test_add_console_handler_stderr(self):
        """Test console handler with stderr stream."""
        config = LoggerConfig(name="test_stderr")
        config.add_console_handler(stream=sys.stderr)

        handler = config.logger.handlers[0]
        assert handler.stream == sys.stderr

    def test_add_file_handler(self, tmp_path):
        """Test adding file handler."""
        log_file = tmp_path / "test.log"
        config = LoggerConfig(name="test_file")
        result = config.add_file_handler(str(log_file))

        assert result is config  # Method chaining
        assert len(config.logger.handlers) == 1
        assert isinstance(config.logger.handlers[0], logging.FileHandler)
        assert log_file.exists()

    def test_add_file_handler_creates_directory(self, tmp_path):
        """Test that file handler creates parent directories."""
        log_file = tmp_path / "subdir" / "nested" / "test.log"
        config = LoggerConfig(name="test_file_dir")
        config.add_file_handler(str(log_file))

        assert log_file.parent.exists()

    def test_add_rotating_file_handler(self, tmp_path):
        """Test adding rotating file handler."""
        log_file = tmp_path / "rotating.log"
        config = LoggerConfig(name="test_rotating")
        result = config.add_rotating_file_handler(
            str(log_file), max_bytes=1024, backup_count=3
        )

        assert result is config  # Method chaining
        assert len(config.logger.handlers) == 1
        handler = config.logger.handlers[0]
        assert isinstance(handler, logging.handlers.RotatingFileHandler)
        assert handler.maxBytes == 1024
        assert handler.backupCount == 3

    def test_add_timed_rotating_handler(self, tmp_path):
        """Test adding timed rotating file handler."""
        log_file = tmp_path / "timed.log"
        config = LoggerConfig(name="test_timed")
        result = config.add_timed_rotating_handler(
            str(log_file), when="midnight", interval=1, backup_count=7
        )

        assert result is config  # Method chaining
        assert len(config.logger.handlers) == 1
        handler = config.logger.handlers[0]
        assert isinstance(handler, logging.handlers.TimedRotatingFileHandler)
        assert handler.when == "MIDNIGHT"
        # TimedRotatingFileHandler converts interval to seconds for 'midnight'
        assert handler.interval == 86400  # 1 day in seconds
        assert handler.backupCount == 7

    def test_multiple_handlers_chaining(self, tmp_path):
        """Test adding multiple handlers via method chaining."""
        log_file = tmp_path / "multi.log"
        config = (
            LoggerConfig(name="test_multi")
            .add_console_handler()
            .add_file_handler(str(log_file))
            .add_rotating_file_handler(str(tmp_path / "rotating.log"))
        )

        assert len(config.logger.handlers) == 3

    def test_get_logger(self):
        """Test getting logger instance."""
        config = LoggerConfig(name="test_get_logger")
        logger = config.get_logger()

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_get_logger"

    def test_logger_level_propagation(self):
        """Test that logger level is set correctly."""
        config = LoggerConfig(name="test_level", level=logging.WARNING)
        logger = config.get_logger()

        assert logger.level == logging.WARNING

    def test_handler_formatting(self, tmp_path, caplog):
        """Test that handlers use the correct formatter."""
        log_file = tmp_path / "format_test.log"
        custom_format = "CUSTOM: %(levelname)s - %(message)s"

        config = LoggerConfig(name="test_format", format_string=custom_format)
        config.add_file_handler(str(log_file))

        logger = config.get_logger()
        logger.info("test message")

        content = log_file.read_text()
        assert "CUSTOM: INFO - test message" in content
