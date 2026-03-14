"""Runtime hook: ensure onnxruntime native DLLs are discoverable in frozen builds.

On Windows, Python 3.8+ no longer searches PATH for DLLs.  onnxruntime's own
__init__.py calls os.add_dll_directory() for its ``capi`` folder, but in a
PyInstaller frozen bundle the directory layout may differ from a normal
site-packages install.  This hook adds the relevant directories early — before
any application code imports onnxruntime.
"""

import os
import sys

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _base = sys._MEIPASS

    _candidate_dirs = [
        os.path.join(_base, "onnxruntime", "capi"),
        _base,
    ]

    for _d in _candidate_dirs:
        if os.path.isdir(_d):
            if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
                os.add_dll_directory(_d)
            # Also prepend to PATH as a fallback for older Windows / edge cases
            os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")
