"""Set up app-local MSVC DLL loading in the active Windows virtual environment.

Run from the repository after installing scripts/policies/sequences/requirements-sequences.txt:
    uv run --no-sync python scripts/policies/sequences/setup_sequences_windows.py

This only writes two startup files inside .venv. It does not change Windows DLLs.
"""

import os
import sys
import sysconfig
from pathlib import Path


def main():
    """Enable the runtime installed by msvc-runtime before importing native packages."""
    if os.name != "nt":
        print("Windows-only setup; no changes needed on this platform")
        return
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run inside the project's virtual environment")
    if not (Path(sys.executable).parent / "msvcp140.dll").is_file():
        raise RuntimeError("Install requirements-sequences.txt first")
    destination = Path(sysconfig.get_path("purelib"))
    module = destination / "lerobot_local_vc_runtime.py"
    module.write_text(
        '''"""Load this environment's MSVC runtime before native packages on Windows."""
import ctypes
import os
import sys
from pathlib import Path
_handles = []
if os.name == "nt":
    _runtime = Path(sys.executable).parent
    for _name in ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "msvcp140_atomic_wait.dll"):
        _path = _runtime / _name
        if _path.is_file():
            _handles.append(ctypes.WinDLL(str(_path)))
''',
        encoding="utf-8",
    )
    (destination / "lerobot_local_vc_runtime.pth").write_text(
        "import lerobot_local_vc_runtime\n", encoding="utf-8"
    )
    print(f"Configured app-local MSVC runtime in {destination}")


if __name__ == "__main__":
    main()
