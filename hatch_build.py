import os
import sys
from pathlib import Path
from subprocess import run
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def clean(self, versions: list[str]) -> None:
        build_dir = Path(self.directory)

        for binary in ("parser", "gls"):
            (build_dir / binary).unlink(missing_ok=True)

        super().clean(versions)

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        self.app.display_waiting("Bootstrapping parser")
        rc = run(["./bootstrap.sh"], check=False).returncode
        if rc:
            self.app.abort("Bootstrap failed", rc)

        self.app.display_waiting("Compiling gls")

        rc = run(
            [
                sys.executable,
                "-m",
                "src.glang.driver",
                "src/gls/server.c3",
                "-o",
                f"{self.directory}/gls",
                "-O3",
            ],
            check=False,
            env=os.environ
            | {
                "PYTHONPATH": str(Path(self.root) / "src")
                + ":"
                + os.environ.get("PYTHONPATH", "")
            },
        ).returncode
        if rc:
            self.app.abort("gls compilation failed")

        self.app.display_success("Done!")

        build_data["force_include"]["dist/parser"] = "glang/bin/parser"
        build_data["force_include"][f"{self.directory}/gls"] = "glang/bin/gls"

        build_data["pure_python"] = False
        build_data["infer_tag"] = True

        super().initialize(version, build_data)
