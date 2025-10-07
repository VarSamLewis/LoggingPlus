"""
Logging Utilities Library
A flexible logging configuration utility with support for multiple handlers,
file persistence, and standard Python logging functions.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, List, Dict, Any
from datetime import datetime


class LoggerConfig:
    """Configuration class for setting up loggers with various handlers."""
    
    def __init__(
        self,
        name: str = __name__,
        level: int = logging.INFO,
        format_string: Optional[str] = None
    ):
        """
        Initialize logger configuration.
        
        Args:
            name: Logger name (default: module name)
            level: Logging level (default: INFO)
            format_string: Custom format string for log messages
        """
        self.name = name
        self.level = level
        self.format_string = format_string or (
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()  # Clear existing handlers
        self.formatter = logging.Formatter(self.format_string)
    
    def add_console_handler(
        self,
        level: Optional[int] = None,
        stream=sys.stdout
    ) -> 'LoggerConfig':
        """
        Add console/stream handler.
        
        Args:
            level: Logging level for this handler (default: logger level)
            stream: Output stream (default: stdout)
        
        Returns:
            self for method chaining
        """
        handler = logging.StreamHandler(stream)
        handler.setLevel(level or self.level)
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)
        return self
    
    def add_file_handler(
        self,
        filepath: str,
        level: Optional[int] = None,
        mode: str = 'a'
    ) -> 'LoggerConfig':
        """
        Add basic file handler.
        
        Args:
            filepath: Path to log file
            level: Logging level for this handler (default: logger level)
            mode: File mode ('a' for append, 'w' for write)
        
        Returns:
            self for method chaining
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(filepath, mode=mode)
        handler.setLevel(level or self.level)
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)
        return self
    
    def add_rotating_file_handler(
        self,
        filepath: str,
        max_bytes: int = 10485760,  # 10MB
        backup_count: int = 5,
        level: Optional[int] = None
    ) -> 'LoggerConfig':
        """
        Add rotating file handler (rotates based on file size).
        
        Args:
            filepath: Path to log file
            max_bytes: Maximum file size before rotation (default: 10MB)
            backup_count: Number of backup files to keep
            level: Logging level for this handler (default: logger level)
        
        Returns:
            self for method chaining
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            filepath,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        handler.setLevel(level or self.level)
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)
        return self
    
    def add_timed_rotating_handler(
        self,
        filepath: str,
        when: str = 'midnight',
        interval: int = 1,
        backup_count: int = 7,
        level: Optional[int] = None
    ) -> 'LoggerConfig':
        """
        Add timed rotating file handler (rotates based on time).
        
        Args:
            filepath: Path to log file
            when: Type of interval ('S', 'M', 'H', 'D', 'midnight', 'W0'-'W6')
            interval: Number of intervals between rotations
            backup_count: Number of backup files to keep
            level: Logging level for this handler (default: logger level)
        
        Returns:
            self for method chaining
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        handler = TimedRotatingFileHandler(
            filepath,
            when=when,
            interval=interval,
            backupCount=backup_count
        )
        handler.setLevel(level or self.level)
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)
        return self
    
    def get_logger(self) -> logging.Logger:
        """
        Get the configured logger instance.
        
        Returns:
            Configured logger object
        """
        return self.logger


class Logger:
    """Wrapper class providing convenient logging methods."""
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize logger wrapper.
        
        Args:
            logger: Python logging.Logger instance
        """
        self._logger = logger
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message."""
        self._logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        self._logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message."""
        self._logger.warning(msg, *args, **kwargs)
    
    def warn(self, msg: str, *args, **kwargs):
        """Alias for warning."""
        self._logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error message."""
        self._logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log critical message."""
        self._logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        """Log exception with traceback."""
        self._logger.exception(msg, *args, **kwargs)
    
    def log(self, level: int, msg: str, *args, **kwargs):
        """Log message at specified level."""
        self._logger.log(level, msg, *args, **kwargs)
    
    def set_level(self, level: int):
        """Change logger level."""
        self._logger.setLevel(level)
    
    @property
    def level(self) -> int:
        """Get current logger level."""
        return self._logger.level
    
class LoggerPersistence:
    """Flexible file persistence utility for logging and data storage."""
    
    def __init__(
        self,
        base_path: str = 'logs',
        logger: Optional[Logger] = None
    ):
        """
        Initialize logger persistence with a base directory.
        
        Args:
            base_path: Base directory for log files (default: 'logs')
            logger: Optional Logger instance to log persistence operations
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger
    
    def _log(self, level: str, msg: str):
        """Internal logging helper."""
        if self.logger:
            getattr(self.logger, level)(msg)
    
    def _resolve_path(self, filepath: str) -> Path:
        """Resolve filepath relative to base_path."""
        path = Path(filepath)
        if not path.is_absolute():
            path = self.base_path / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    def write_text(
        self,
        content: str,
        filepath: str,
        mode: str = 'w',
        encoding: str = 'utf-8'
    ) -> Path:
        """
        Write text content to file.
        
        Args:
            content: Text content to write
            filepath: Path to file (relative to base_path or absolute)
            mode: Write mode ('w', 'a', etc.)
            encoding: File encoding (default: 'utf-8')
        
        Returns:
            Path to written file
        """
        path = self._resolve_path(filepath)
        try:
            path.write_text(content, encoding=encoding)
            self._log('info', f"Written text to {path}")
            return path
        except Exception as e:
            self._log('error', f"Failed to write text to {path}: {e}")
            raise
    
    def write_json(
        self,
        data: Any,
        filepath: str,
        indent: Optional[int] = 2,
        mode: str = 'w'
    ) -> Path:
        """
        Write data as JSON to file.
        
        Args:
            data: Data to serialize as JSON
            filepath: Path to file (relative to base_path or absolute)
            indent: JSON indentation (default: 2, None for compact)
            mode: Write mode ('w' or 'a')
        
        Returns:
            Path to written file
        """
        import json
        
        path = self._resolve_path(filepath)
        try:
            with open(path, mode, encoding='utf-8') as f:
                json.dump(data, f, indent=indent, default=str)
            self._log('info', f"Written JSON to {path}")
            return path
        except Exception as e:
            self._log('error', f"Failed to write JSON to {path}: {e}")
            raise
    
    def write_csv(
        self,
        data: List[Dict],
        filepath: str,
        mode: str = 'w',
        fieldnames: Optional[List[str]] = None
    ) -> Path:
        """
        Write data as CSV to file.
        
        Args:
            data: List of dictionaries to write
            filepath: Path to file (relative to base_path or absolute)
            mode: Write mode ('w' or 'a')
            fieldnames: CSV column names (auto-detected if None)
        
        Returns:
            Path to written file
        """
        import csv
        
        if not data:
            raise ValueError("Cannot write empty data to CSV")
        
        path = self._resolve_path(filepath)
        
        if fieldnames is None:
            fieldnames = list(data[0].keys())
        
        try:
            with open(path, mode, newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if mode == 'w' or not path.exists():
                    writer.writeheader()
                writer.writerows(data)
            self._log('info', f"Written CSV to {path}")
            return path
        except Exception as e:
            self._log('error', f"Failed to write CSV to {path}: {e}")
            raise
    
    def write_binary(
        self,
        data: bytes,
        filepath: str,
        mode: str = 'wb'
    ) -> Path:
        """
        Write binary data to file.
        
        Args:
            data: Binary data to write
            filepath: Path to file (relative to base_path or absolute)
            mode: Write mode ('wb' or 'ab')
        
        Returns:
            Path to written file
        """
        path = self._resolve_path(filepath)
        try:
            path.write_bytes(data)
            self._log('info', f"Written binary data to {path}")
            return path
        except Exception as e:
            self._log('error', f"Failed to write binary to {path}: {e}")
            raise
    
    def append_line(
        self,
        line: str,
        filepath: str,
        add_newline: bool = True
    ) -> Path:
        """
        Append a single line to file.
        
        Args:
            line: Line to append
            filepath: Path to file (relative to base_path or absolute)
            add_newline: Add newline character if not present
        
        Returns:
            Path to written file
        """
        if add_newline and not line.endswith('\n'):
            line += '\n'
        return self.write_text(line, filepath, mode='a')
    
    def write_pickle(
        self,
        data: Any,
        filepath: str
    ) -> Path:
        """
        Write data using pickle serialization.
        
        Args:
            data: Data to pickle
            filepath: Path to file (relative to base_path or absolute)
        
        Returns:
            Path to written file
        """
        import pickle
        
        path = self._resolve_path(filepath)
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            self._log('info', f"Written pickle to {path}")
            return path
        except Exception as e:
            self._log('error', f"Failed to write pickle to {path}: {e}")
            raise
    
    def read_text(
        self,
        filepath: str,
        encoding: str = 'utf-8'
    ) -> str:
        """
        Read text from file.
        
        Args:
            filepath: Path to file (relative to base_path or absolute)
            encoding: File encoding (default: 'utf-8')
        
        Returns:
            File contents as string
        """
        path = self._resolve_path(filepath)
        try:
            content = path.read_text(encoding=encoding)
            self._log('debug', f"Read text from {path}")
            return content
        except Exception as e:
            self._log('error', f"Failed to read text from {path}: {e}")
            raise
    
    def read_json(self, filepath: str) -> Any:
        """
        Read JSON from file.
        
        Args:
            filepath: Path to file (relative to base_path or absolute)
        
        Returns:
            Deserialized JSON data
        """
        import json
        
        path = self._resolve_path(filepath)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._log('debug', f"Read JSON from {path}")
            return data
        except Exception as e:
            self._log('error', f"Failed to read JSON from {path}: {e}")
            raise
    
    def read_csv(
        self,
        filepath: str,
        as_dict: bool = True
    ) -> List:
        """
        Read CSV from file.
        
        Args:
            filepath: Path to file (relative to base_path or absolute)
            as_dict: Return as list of dicts (True) or list of lists (False)
        
        Returns:
            CSV data
        """
        import csv
        
        path = self._resolve_path(filepath)
        try:
            with open(path, 'r', newline='', encoding='utf-8') as f:
                if as_dict:
                    data = list(csv.DictReader(f))
                else:
                    data = list(csv.reader(f))
            self._log('debug', f"Read CSV from {path}")
            return data
        except Exception as e:
            self._log('error', f"Failed to read CSV from {path}: {e}")
            raise
    
    def read_binary(self, filepath: str) -> bytes:
        """
        Read binary data from file.
        
        Args:
            filepath: Path to file (relative to base_path or absolute)
        
        Returns:
            Binary data
        """
        path = self._resolve_path(filepath)
        try:
            data = path.read_bytes()
            self._log('debug', f"Read binary from {path}")
            return data
        except Exception as e:
            self._log('error', f"Failed to read binary from {path}: {e}")
            raise
    
    def read_pickle(self, filepath: str) -> Any:
        """
        Read pickled data from file.
        
        Args:
            filepath: Path to file (relative to base_path or absolute)
        
        Returns:
            Unpickled data
        """
        import pickle
        
        path = self._resolve_path(filepath)
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self._log('debug', f"Read pickle from {path}")
            return data
        except Exception as e:
            self._log('error', f"Failed to read pickle from {path}: {e}")
            raise
    
    def list_files(
        self,
        pattern: str = "*",
        recursive: bool = False
    ) -> List[Path]:
        """
        List files in base_path matching pattern.
        
        Args:
            pattern: Glob pattern (default: "*")
            recursive: Search recursively (default: False)
        
        Returns:
            List of matching file paths
        """
        if recursive:
            files = list(self.base_path.rglob(pattern))
        else:
            files = list(self.base_path.glob(pattern))
        return [f for f in files if f.is_file()]
    
    def delete_file(self, filepath: str) -> bool:
        """
        Delete a file.
        
        Args:
            filepath: Path to file (relative to base_path or absolute)
        
        Returns:
            True if deleted, False if file didn't exist
        """
        path = self._resolve_path(filepath)
        try:
            if path.exists():
                path.unlink()
                self._log('info', f"Deleted file {path}")
                return True
            return False
        except Exception as e:
            self._log('error', f"Failed to delete {path}: {e}")
            raise


def setup_logger(
    name: str = __name__,
    level: int = logging.INFO,
    handlers: Optional[List[Dict[str, Any]]] = None,
    format_string: Optional[str] = None
) -> Logger:
    """
    Quick setup function for common logging configurations.
    
    Args:
        name: Logger name
        level: Logging level
        handlers: List of handler configurations
        format_string: Custom format string
    
    Returns:
        Logger instance
    
    Example:
        logger = setup_logger(
            name='my_app',
            level=logging.DEBUG,
            handlers=[
                {'type': 'console'},
                {'type': 'file', 'filepath': 'logs/app.log'},
                {'type': 'rotating', 'filepath': 'logs/rotating.log', 'max_bytes': 5242880}
            ]
        )
    """
    config = LoggerConfig(name=name, level=level, format_string=format_string)
    
    if handlers is None:
        handlers = [{'type': 'console'}]
    
    for handler_config in handlers:
        handler_type = handler_config.pop('type')
        
        if handler_type == 'console':
            config.add_console_handler(**handler_config)
        elif handler_type == 'file':
            config.add_file_handler(**handler_config)
        elif handler_type == 'rotating':
            config.add_rotating_file_handler(**handler_config)
        elif handler_type == 'timed':
            config.add_timed_rotating_handler(**handler_config)
        else:
            raise ValueError(f"Unknown handler type: {handler_type}")
    
    return Logger(config.get_logger())


# Example usage
if __name__ == "__main__":
    # Example 1: Simple console logger
    logger1 = setup_logger(name='simple_app', level=logging.INFO)
    logger1.info("This is an info message")
    logger1.debug("This debug won't show (level is INFO)")
    
    # Example 2: Multiple handlers with method chaining
    config = LoggerConfig(name='multi_app', level=logging.DEBUG)
    logger2 = Logger(
        config
        .add_console_handler()
        .add_file_handler('logs/app.log')
        .add_rotating_file_handler(
            'logs/rotating.log',
            max_bytes=5242880,  # 5MB
            backup_count=3
        )
        .get_logger()
    )
    
    logger2.debug("Debug message")
    logger2.info("Info message")
    logger2.warning("Warning message")
    logger2.error("Error message")
    
    # Example 3: Quick setup with configuration dict
    logger3 = setup_logger(
        name='configured_app',
        level=logging.INFO,
        handlers=[
            {'type': 'console', 'level': logging.WARNING},
            {'type': 'file', 'filepath': 'logs/all.log'},
            {'type': 'rotating', 'filepath': 'logs/rotating.log', 'max_bytes': 1048576, 'backup_count': 5},
            {'type': 'timed', 'filepath': 'logs/daily.log', 'when': 'midnight', 'backup_count': 7}
        ],
        format_string='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )
    
    logger3.info("Application started")
    logger3.warning("This is a warning")
    
    try:
        1 / 0
    except Exception as e:
        logger3.exception("An error occurred")