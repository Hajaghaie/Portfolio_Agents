"""
Configuration and environment variable loading, Put you api keys here, and other configuration variables. 
preference is to use .env file to store variables.
"""
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Example constants (move more as needed)
BENCHMARK_TICKER = "^GSPC"
DEFAULT_PERIOD = "5y"

# Fetch API keys (if needed elsewhere)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY not set in environment.")
if not TAVILY_API_KEY:
    logging.warning("TAVILY_API_KEY not set in environment.") 