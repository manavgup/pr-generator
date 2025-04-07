#!/usr/bin/env python
"""
Runner script for the PR Recommendation System.
"""
import sys
from shared.utils.logging_utils import get_logger
from crewai_approach.main import main as main_func

logger = get_logger(__name__)


def main():
    """Main function for running the PR recommendation system."""
    return main_func()


if __name__ == "__main__":
    sys.exit(main())