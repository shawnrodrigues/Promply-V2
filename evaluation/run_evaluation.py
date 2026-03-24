#!/usr/bin/env python3
"""
Promply-V2 Evaluation Launcher
Fixes sys.path to enable direct execution from project root
"""

import sys
from pathlib import Path

# Add parent directory to sys.path for relative imports
parent_dir = Path(__file__).parent / 'src'
sys.path.insert(0, str(parent_dir))

from run_evaluation import main

if __name__ == '__main__':
    main()
