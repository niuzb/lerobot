"""Start the tested Isaac runtime and expose the isolated Lab and LeIsaac APIs."""

import builtins
import os
import sys
from pathlib import Path


def launch(run, leisaac_root, cameras=False):
    import site

    packages = Path(run) / "site-packages"
    site.addsitedir(str(packages))
    # The PyPI distribution contains the extension sources under this directory.
    lab_source = packages / "isaaclab/source"
    for name in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_mimic", "isaaclab_rl"):
        sys.path.insert(0, str(lab_source / name))
    sys.path.insert(0, str(Path(leisaac_root) / "source/leisaac"))
    builtins.ISAAC_LAUNCHED_FROM_TERMINAL = True

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "enable_crashreporter": False,
            "limit_cpu_threads": int(os.environ.get("LEISAAC_CPU_THREADS", "4")),
            "multi_gpu": False,
            "active_gpu": 0,
            "anti_aliasing": 2,
            "denoiser": False,
            "extra_args": ["--allow-root", "--/rtx/verifyDriverVersion/enabled=false"],
        }
    )
    import carb.settings

    settings = carb.settings.get_settings()
    settings.set("/isaaclab/cameras_enabled", bool(cameras))
    settings.set("/rtx-transient/dldenoiser/enabled", False)
    settings.set("/rtx-transient/dlssg/enabled", False)
    return app
