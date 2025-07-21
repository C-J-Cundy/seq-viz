#!/usr/bin/env python3
"""Convenience script to run the visualization server."""

import sys
import os

# Add the parent directory to the path to avoid module warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seq_viz.server.file_visualization_server import main

if __name__ == "__main__":
    main()