import subprocess

from pathlib import Path
from typing import Any

PARSER_PATH = Path(__file__).parent / "parser"


def parse(path: Path) -> str:
    return subprocess.check_output([PARSER_PATH, path], encoding="utf8")
