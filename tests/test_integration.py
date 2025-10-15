"""
Integration tests for the logging utility library
"""

import logging
import pytest
from pathlib import Path
from src.loggingplus import setup_logger, LoggerPersistence


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_logger_with_persistence(self, tmp_path):
        """Test using logger together with persistence."""
        # Setup logger
        log_file = tmp_path / "app.log"
        logger = setup_logger(
            name="integrated_app",
            level=logging.INFO,
            handlers=[{"type": "file", "filepath": str(log_file)}],
        )

        # Setup persistence with logger
        persistence = LoggerPersistence(
            base_path=str(tmp_path / "data"), logger=logger
        )

        # Perform operations
        data = {"user": "Alice", "action": "login"}
        persistence.write_json(data, "user_data.json")

        # Verify logger captured persistence operations
        log_content = log_file.read_text()
        assert "Written JSON to" in log_content

        # Verify data was written correctly
        loaded_data = persistence.read_json("user_data.json")
        assert loaded_data == data

    def test_multi_handler_logging(self, tmp_path):
        """Test logging to multiple handlers simultaneously."""
        console_log = tmp_path / "console.log"
        error_log = tmp_path / "errors.log"
        rotating_log = tmp_path / "rotating.log"

        logger = setup_logger(
            name="multi_handler_app",
            level=logging.DEBUG,
            handlers=[
                {"type": "file", "filepath": str(console_log), "level": logging.DEBUG},
                {"type": "file", "filepath": str(error_log), "level": logging.ERROR},
                {
                    "type": "rotating",
                    "filepath": str(rotating_log),
                    "max_bytes": 1024,
                    "backup_count": 3,
                },
            ],
        )

        # Log various levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Verify console log has everything
        console_content = console_log.read_text()
        assert "Debug message" in console_content
        assert "Info message" in console_content
        assert "Warning message" in console_content
        assert "Error message" in console_content

        # Verify error log has only errors
        error_content = error_log.read_text()
        assert "Debug message" not in error_content
        assert "Info message" not in error_content
        assert "Warning message" not in error_content
        assert "Error message" in error_content

        # Verify rotating log has everything
        rotating_content = rotating_log.read_text()
        assert "Debug message" in rotating_content
        assert "Error message" in rotating_content

    def test_exception_logging_with_context(self, tmp_path):
        """Test exception logging with full context."""
        log_file = tmp_path / "exceptions.log"
        logger = setup_logger(
            name="exception_app",
            level=logging.DEBUG,
            handlers=[{"type": "file", "filepath": str(log_file)}],
        )

        def nested_function():
            """A nested function that raises an exception."""
            raise ValueError("Something went wrong")

        def outer_function():
            """An outer function that calls the nested function."""
            try:
                nested_function()
            except ValueError:
                logger.exception("Caught exception in outer_function")

        # Execute
        outer_function()

        # Verify full traceback is logged
        content = log_file.read_text()
        assert "ERROR" in content
        assert "Caught exception in outer_function" in content
        assert "ValueError: Something went wrong" in content
        assert "Traceback" in content
        assert "nested_function" in content

    def test_persistence_multiple_formats(self, tmp_path):
        """Test writing and reading multiple file formats."""
        persistence = LoggerPersistence(base_path=str(tmp_path))

        # Test data
        text_data = "This is plain text\nWith multiple lines\n"
        json_data = {"users": ["Alice", "Bob"], "count": 2}
        csv_data = [
            {"name": "Alice", "score": 95},
            {"name": "Bob", "score": 87},
        ]
        binary_data = b"\x00\x01\x02\x03\x04\x05"

        # Write all formats
        persistence.write_text(text_data, "data.txt")
        persistence.write_json(json_data, "data.json")
        persistence.write_csv(csv_data, "data.csv")
        persistence.write_binary(binary_data, "data.bin")

        # Read all formats
        read_text = persistence.read_text("data.txt")
        read_json = persistence.read_json("data.json")
        read_csv = persistence.read_csv("data.csv")
        read_binary = persistence.read_binary("data.bin")

        # Verify
        assert read_text == text_data
        assert read_json == json_data
        assert len(read_csv) == 2
        assert read_csv[0]["name"] == "Alice"
        assert read_binary == binary_data

    def test_rotating_log_with_size_limit(self, tmp_path):
        """Test that rotating file handler actually rotates."""
        log_file = tmp_path / "rotating.log"

        logger = setup_logger(
            name="rotation_test",
            level=logging.INFO,
            handlers=[
                {
                    "type": "rotating",
                    "filepath": str(log_file),
                    "max_bytes": 500,  # Small size to trigger rotation
                    "backup_count": 3,
                }
            ],
        )

        # Write enough logs to trigger rotation
        for i in range(100):
            logger.info(f"Log message number {i} with some padding text to increase size")

        # Check that rotation occurred
        # Main log file should exist
        assert log_file.exists()

        # Backup files should exist
        backup1 = Path(str(log_file) + ".1")
        assert backup1.exists()

    def test_csv_append_workflow(self, tmp_path):
        """Test realistic CSV append workflow."""
        persistence = LoggerPersistence(base_path=str(tmp_path))

        # Initial data write
        batch1 = [
            {"timestamp": "2025-01-01 10:00", "user": "Alice", "action": "login"},
            {"timestamp": "2025-01-01 10:05", "user": "Bob", "action": "login"},
        ]
        persistence.write_csv(batch1, "events.csv", mode="w")

        # Append more data
        batch2 = [
            {"timestamp": "2025-01-01 10:10", "user": "Alice", "action": "logout"},
            {"timestamp": "2025-01-01 10:15", "user": "Bob", "action": "logout"},
        ]
        persistence.write_csv(batch2, "events.csv", mode="a")

        # Read and verify
        all_events = persistence.read_csv("events.csv")
        assert len(all_events) == 4
        assert all_events[0]["user"] == "Alice"
        assert all_events[3]["action"] == "logout"

    def test_logger_persistence_file_listing(self, tmp_path):
        """Test listing and managing persisted files."""
        persistence = LoggerPersistence(base_path=str(tmp_path))

        # Create various files
        persistence.write_text("log 1", "app1.log")
        persistence.write_text("log 2", "app2.log")
        persistence.write_json({"data": 1}, "data.json")
        persistence.write_text("config", "subdir/config.txt")

        # List all files
        all_files = persistence.list_files(recursive=True)
        assert len(all_files) == 4

        # List only log files
        log_files = persistence.list_files(pattern="*.log", recursive=False)
        assert len(log_files) == 2

        # Delete a file
        result = persistence.delete_file("app1.log")
        assert result is True

        # Verify deletion
        remaining_logs = persistence.list_files(pattern="*.log", recursive=False)
        assert len(remaining_logs) == 1

    def test_real_world_application_scenario(self, tmp_path):
        """Test a realistic application scenario with logging and persistence."""
        # Setup application logger
        app_log = tmp_path / "app.log"
        error_log = tmp_path / "errors.log"

        logger = setup_logger(
            name="my_application",
            level=logging.INFO,
            handlers=[
                {"type": "file", "filepath": str(app_log)},
                {"type": "file", "filepath": str(error_log), "level": logging.ERROR},
                {
                    "type": "rotating",
                    "filepath": str(tmp_path / "rotating.log"),
                    "max_bytes": 10485760,
                    "backup_count": 5,
                },
            ],
        )

        # Setup data persistence
        persistence = LoggerPersistence(
            base_path=str(tmp_path / "data"), logger=logger
        )

        # Simulate application workflow
        logger.info("Application started")

        # Process some data
        user_data = {"user_id": 123, "username": "alice", "email": "alice@example.com"}
        persistence.write_json(user_data, "users/alice.json")
        logger.info(f"Created user profile for {user_data['username']}")

        # Log some activity
        activity_log = [
            {"timestamp": "2025-01-01 10:00", "action": "login", "user_id": 123},
            {"timestamp": "2025-01-01 10:05", "action": "view_page", "user_id": 123},
        ]
        persistence.write_csv(activity_log, "activity.csv")
        logger.info(f"Logged {len(activity_log)} activities")

        # Simulate an error
        try:
            raise RuntimeError("Database connection failed")
        except RuntimeError:
            logger.exception("Critical error occurred")

        logger.info("Application shutting down")

        # Verify logs
        app_content = app_log.read_text()
        assert "Application started" in app_content
        assert "Created user profile" in app_content
        assert "Critical error occurred" in app_content
        assert "Application shutting down" in app_content

        error_content = error_log.read_text()
        assert "Critical error occurred" in error_content
        assert "Database connection failed" in error_content
        assert "Application started" not in error_content  # Info not in error log

        # Verify persistence
        saved_user = persistence.read_json("users/alice.json")
        assert saved_user["username"] == "alice"

        saved_activity = persistence.read_csv("activity.csv")
        assert len(saved_activity) == 2
