"""
Tests for LoggerPersistence class
"""

import json
import pickle
from pathlib import Path
import pytest
from src.loggingplus import LoggerPersistence, Logger, LoggerConfig


class TestLoggerPersistence:
    """Test cases for LoggerPersistence class."""

    @pytest.fixture
    def persistence(self, tmp_path):
        """Create a LoggerPersistence instance with temp directory."""
        return LoggerPersistence(base_path=str(tmp_path))

    @pytest.fixture
    def persistence_with_logger(self, tmp_path):
        """Create a LoggerPersistence instance with logger."""
        log_file = tmp_path / "persist.log"
        config = LoggerConfig(name="persistence_logger")
        config.add_file_handler(str(log_file))
        logger = Logger(config.get_logger())
        persistence = LoggerPersistence(base_path=str(tmp_path / "data"), logger=logger)
        return persistence, log_file

    def test_init_default(self):
        """Test initialization with default parameters."""
        persistence = LoggerPersistence()
        assert persistence.base_path == Path("logs")
        assert persistence.logger is None
        assert persistence.base_path.exists()

    def test_init_custom_base_path(self, tmp_path):
        """Test initialization with custom base path."""
        custom_path = tmp_path / "custom_logs"
        persistence = LoggerPersistence(base_path=str(custom_path))
        assert persistence.base_path == custom_path
        assert custom_path.exists()

    def test_init_with_logger(self, tmp_path):
        """Test initialization with logger."""
        config = LoggerConfig(name="test")
        logger = Logger(config.get_logger())
        persistence = LoggerPersistence(logger=logger)
        assert persistence.logger is logger

    def test_resolve_path_relative(self, persistence, tmp_path):
        """Test resolving relative paths."""
        path = persistence._resolve_path("test.log")
        assert path == tmp_path / "test.log"
        assert path.parent.exists()

    def test_resolve_path_absolute(self, persistence, tmp_path):
        """Test resolving absolute paths."""
        abs_path = tmp_path / "absolute" / "test.log"
        path = persistence._resolve_path(str(abs_path))
        assert path == abs_path
        assert path.parent.exists()

    def test_resolve_path_nested(self, persistence, tmp_path):
        """Test resolving nested paths."""
        path = persistence._resolve_path("subdir/nested/file.log")
        assert path == tmp_path / "subdir" / "nested" / "file.log"
        assert path.parent.exists()

    def test_write_text(self, persistence, tmp_path):
        """Test writing text to file."""
        content = "Hello, World!"
        path = persistence.write_text(content, "test.txt")

        assert path.exists()
        assert path.read_text() == content

    def test_write_text_append_mode(self, persistence, tmp_path):
        """Test writing text in append mode."""
        persistence.write_text("First line\n", "append.txt", mode="w")
        persistence.write_text("Second line\n", "append.txt", mode="a")

        content = (tmp_path / "append.txt").read_text()
        assert "First line" in content
        assert "Second line" in content

    def test_write_text_custom_encoding(self, persistence, tmp_path):
        """Test writing text with custom encoding."""
        content = "Unicode: \u2713 \u4e2d\u6587"
        path = persistence.write_text(content, "unicode.txt", encoding="utf-8")

        assert path.read_text(encoding="utf-8") == content

    def test_write_json(self, persistence, tmp_path):
        """Test writing JSON data."""
        data = {"name": "John", "age": 30, "active": True}
        path = persistence.write_json(data, "data.json")

        assert path.exists()
        with open(path, "r") as f:
            loaded = json.load(f)
        assert loaded == data

    def test_write_json_compact(self, persistence, tmp_path):
        """Test writing compact JSON."""
        data = {"a": 1, "b": 2}
        path = persistence.write_json(data, "compact.json", indent=None)

        content = path.read_text()
        assert "\n" not in content.strip()

    def test_write_json_with_datetime(self, persistence, tmp_path):
        """Test writing JSON with datetime (using default=str)."""
        from datetime import datetime

        data = {"timestamp": datetime(2025, 1, 1, 12, 0, 0)}
        path = persistence.write_json(data, "datetime.json")

        with open(path, "r") as f:
            loaded = json.load(f)
        assert "2025-01-01" in loaded["timestamp"]

    def test_write_csv(self, persistence, tmp_path):
        """Test writing CSV data."""
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
        ]
        path = persistence.write_csv(data, "data.csv")

        content = path.read_text()
        assert "name,age" in content
        assert "Alice,30" in content
        assert "Bob,25" in content

    def test_write_csv_custom_fieldnames(self, persistence, tmp_path):
        """Test writing CSV with custom fieldnames."""
        data = [{"name": "Alice", "age": 30, "extra": "ignored"}]
        path = persistence.write_csv(data, "custom.csv", fieldnames=["name", "age"])

        content = path.read_text()
        assert "name,age" in content
        assert "extra" not in content

    def test_write_csv_empty_data(self, persistence):
        """Test writing empty CSV raises error."""
        with pytest.raises(ValueError, match="Cannot write empty data"):
            persistence.write_csv([], "empty.csv")

    def test_write_csv_append_mode(self, persistence, tmp_path):
        """Test appending to CSV file."""
        data1 = [{"name": "Alice", "age": 30}]
        data2 = [{"name": "Bob", "age": 25}]

        persistence.write_csv(data1, "append.csv", mode="w")
        persistence.write_csv(data2, "append.csv", mode="a")

        content = (tmp_path / "append.csv").read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 3  # header + 2 data rows
        assert "Alice" in content
        assert "Bob" in content

    def test_write_binary(self, persistence, tmp_path):
        """Test writing binary data."""
        data = b"\x00\x01\x02\x03\x04"
        path = persistence.write_binary(data, "binary.bin")

        assert path.exists()
        assert path.read_bytes() == data

    def test_append_line(self, persistence, tmp_path):
        """Test appending a line to file."""
        persistence.append_line("Line 1", "lines.txt")
        persistence.append_line("Line 2", "lines.txt")

        content = (tmp_path / "lines.txt").read_text()
        assert content == "Line 1\nLine 2\n"

    def test_append_line_no_newline(self, persistence, tmp_path):
        """Test appending line without adding newline."""
        persistence.append_line("No newline", "lines.txt", add_newline=False)
        persistence.append_line("With newline", "lines.txt", add_newline=True)

        content = (tmp_path / "lines.txt").read_text()
        assert content == "No newlineWith newline\n"

    def test_write_pickle(self, persistence, tmp_path):
        """Test writing pickled data."""
        data = {"list": [1, 2, 3], "dict": {"nested": True}}
        path = persistence.write_pickle(data, "data.pkl")

        with open(path, "rb") as f:
            loaded = pickle.load(f)
        assert loaded == data

    def test_read_text(self, persistence, tmp_path):
        """Test reading text from file."""
        file_path = tmp_path / "read.txt"
        file_path.write_text("Test content")

        content = persistence.read_text("read.txt")
        assert content == "Test content"

    def test_read_json(self, persistence, tmp_path):
        """Test reading JSON from file."""
        data = {"key": "value", "number": 42}
        file_path = tmp_path / "read.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        loaded = persistence.read_json("read.json")
        assert loaded == data

    def test_read_csv_as_dict(self, persistence, tmp_path):
        """Test reading CSV as list of dicts."""
        file_path = tmp_path / "read.csv"
        file_path.write_text("name,age\nAlice,30\nBob,25\n")

        data = persistence.read_csv("read.csv", as_dict=True)
        assert len(data) == 2
        assert data[0]["name"] == "Alice"
        assert data[0]["age"] == "30"

    def test_read_csv_as_list(self, persistence, tmp_path):
        """Test reading CSV as list of lists."""
        file_path = tmp_path / "read.csv"
        file_path.write_text("name,age\nAlice,30\n")

        data = persistence.read_csv("read.csv", as_dict=False)
        assert len(data) == 2  # header + 1 data row
        assert data[0] == ["name", "age"]
        assert data[1] == ["Alice", "30"]

    def test_read_binary(self, persistence, tmp_path):
        """Test reading binary data."""
        data = b"\x00\x01\x02\x03"
        file_path = tmp_path / "read.bin"
        file_path.write_bytes(data)

        loaded = persistence.read_binary("read.bin")
        assert loaded == data

    def test_read_pickle(self, persistence, tmp_path):
        """Test reading pickled data."""
        data = {"test": [1, 2, 3]}
        file_path = tmp_path / "read.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(data, f)

        loaded = persistence.read_pickle("read.pkl")
        assert loaded == data

    def test_list_files(self, persistence, tmp_path):
        """Test listing files in directory."""
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.log").touch()
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file3.txt").touch()

        files = persistence.list_files()
        file_names = [f.name for f in files]
        assert "file1.txt" in file_names
        assert "file2.log" in file_names
        assert "file3.txt" not in file_names  # Not recursive

    def test_list_files_pattern(self, persistence, tmp_path):
        """Test listing files with pattern."""
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.log").touch()

        files = persistence.list_files(pattern="*.txt")
        file_names = [f.name for f in files]
        assert "file1.txt" in file_names
        assert "file2.log" not in file_names

    def test_list_files_recursive(self, persistence, tmp_path):
        """Test listing files recursively."""
        (tmp_path / "file1.txt").touch()
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "file2.txt").touch()

        files = persistence.list_files(pattern="*.txt", recursive=True)
        assert len(files) == 2

    def test_delete_file(self, persistence, tmp_path):
        """Test deleting a file."""
        file_path = tmp_path / "delete.txt"
        file_path.touch()

        result = persistence.delete_file("delete.txt")
        assert result is True
        assert not file_path.exists()

    def test_delete_nonexistent_file(self, persistence):
        """Test deleting nonexistent file returns False."""
        result = persistence.delete_file("nonexistent.txt")
        assert result is False

    def test_error_logging(self, persistence_with_logger):
        """Test that errors are logged."""
        persistence, log_file = persistence_with_logger

        try:
            persistence.read_text("nonexistent.txt")
        except Exception:
            pass

        log_content = log_file.read_text()
        assert "Failed to read text from" in log_content

    def test_write_text_error_handling(self, persistence, tmp_path):
        """Test write_text error handling."""
        # Try to write to invalid path (if on Windows, use reserved name)
        with pytest.raises(Exception):
            # Create a directory with the target name to cause error
            (tmp_path / "exists_as_dir").mkdir()
            persistence.write_text("test", "exists_as_dir")

    def test_read_nonexistent_file_raises(self, persistence):
        """Test reading nonexistent file raises error."""
        with pytest.raises(Exception):
            persistence.read_text("nonexistent.txt")

    def test_read_json_invalid_raises(self, persistence, tmp_path):
        """Test reading invalid JSON raises error."""
        file_path = tmp_path / "invalid.json"
        file_path.write_text("not valid json{")

        with pytest.raises(Exception):
            persistence.read_json("invalid.json")
