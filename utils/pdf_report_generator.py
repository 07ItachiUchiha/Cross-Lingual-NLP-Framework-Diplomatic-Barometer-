"""
[DEPRECATED FOR RESEARCH THESIS SCOPE]
This module has been archived in archive/legacy_enterprise/pdf_report_generator.py.
Retained here as a lightweight compatibility shim so existing test suites and imports remain functional.
"""

import sys
from pathlib import Path

# Try importing the full archived implementation if present
try:
    from archive.legacy_enterprise.pdf_report_generator import (
        PDFReportGenerator,
        REPORTLAB_AVAILABLE,
    )
except (ImportError, ModuleNotFoundError):
    REPORTLAB_AVAILABLE = False

    class PDFReportGenerator:
        """Lightweight deprecated shim"""
        def __init__(self, output_dir=None):
            self.output_dir = output_dir

        def generate_report(self, *args, **kwargs):
            raise NotImplementedError("PDFReportGenerator has been archived for research thesis scope.")
