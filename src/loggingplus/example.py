from core import loggingplus

# Basic instantiation
logger = loggingplus.StructuredLogger("my_app")
logger.info("Simple message")
logger.info("Structured message", user_id=123, action="login")
