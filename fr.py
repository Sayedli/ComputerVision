#!/usr/bin/env python3
"""
Face Recognition MVP (CLI entry)

This file is now a thin wrapper around the `fr` package which
contains the implementation split into modules for clarity.
"""

from __future__ import annotations

from fr.cli import main


if __name__ == "__main__":
    main()

