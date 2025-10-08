#!/usr/bin/env python3
"""
CLI wrapper for the FastReID config generator.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_config import main

if __name__ == "__main__":
    main()
