from pathlib import Path
from subprocess import run
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def clean(self, versions: list[str]):
        build_dir = Path(self.directory)
        parser_bin = build_dir / "parser"

        if parser_bin.is_file():
            parser_bin.unlink()

        return super().clean(versions)

    def initialize(self, version: str, build_data: dict[str, Any]):
        self.app.display_waiting("Bootstrapping parser")

        rc = run(["./bootstrap.sh"], check=False).returncode
        if rc:
            self.app.abort("Bootstrap failed", rc)

        self.app.display_success("Done!")

        build_data["force_include"]["dist/parser"] = "glang/bin/parser"

        build_data["pure_python"] = False
        build_data["infer_tag"] = True

        return super().initialize(version, build_data)
