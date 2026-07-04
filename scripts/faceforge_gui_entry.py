"""PyInstaller GUI entry point for OOTP FaceForge."""
from __future__ import annotations

import multiprocessing

from ootp_faceforge.gui import main


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
