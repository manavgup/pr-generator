# llm_guided_approach/main.py
#!/usr/bin/env python3
"""
Main entry point for the LLM-guided approach to PR generation.
"""
import sys
from llm_guided_approach.scripts.generate_prs import main

if __name__ == "__main__":
    sys.exit(main())