from pathlib import Path
from sysconfig import get_platform
from typing import Any

from hatch.utils.platform import Platform
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def get_build_tag() -> str:
    # Based on https://peps.python.org/pep-0425/. We could also use infer_tag
    # (https://hatch.pypa.io/latest/plugins/builder/wheel/#build-data),
    # but that is a bit too specific (we don't really about the Python ABI).
    python_tag = "py3"  # Generic Python.
    abi_tag = "none"  # We don't use extension modules.
    platform_tag = get_platform().replace("-", "_").replace(".", "_")

    return f"{python_tag}-{abi_tag}-{platform_tag}"


class CustomBuildHook(BuildHookInterface):
    def clean(self, versions: list[str]):
        build_dir = Path(self.directory)
        parser_bin = build_dir / "parser"

        if parser_bin.is_file():
            parser_bin.unlink()

        return super().clean(versions)

    def initialize(self, version: str, build_data: dict[str, Any]):
        self.app.display_waiting("Bootstrapping parser")

        Platform().check_command(["./bootstrap.sh"], shell=False)

        self.app.display_success("Done!")

        build_data["force_include"]["dist/parser"] = "glang/bin/parser"

        build_data["pure_python"] = False
        build_data["tag"] = get_build_tag()

        return super().initialize(version, build_data)
