#!/usr/bin/env python3
"""
Admission Control & TPU Serving Suite Entrypoint.
Delegates to `admission_control.cli.main()`.
"""

import sys
from admission_control.cli import main

if __name__ == "__main__":
    main()
