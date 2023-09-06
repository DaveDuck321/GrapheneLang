import json
import subprocess
from importlib import resources
from pathlib import Path

from ..codegen.user_facing_errors import InvalidSyntax


def is_whitespace(string: str) -> bool:
    return len(string) == 0 or string.isspace()


def parse(path: Path) -> str:
    with resources.as_file(resources.files("glang") / "bin" / "parser") as parser_path:
        result = subprocess.run(
            [parser_path, path],
            encoding="utf8",
            capture_output=True,
            check=False,
        )

    if result.returncode == 0:
        # Success!
        return result.stdout

    # The parser failed, display the error
    error = json.loads(result.stdout)
    assert error["success"] is False

    line = error["location"]["line"]
    column = error["location"]["column"]

    file_lines = path.open().readlines()

    context = file_lines[line - 1 : line]
    if is_whitespace(context[-1][: column - 1]):
        context = file_lines[max(line - 2, 0) : line]

    context.append(" " * (column - 1) + "^")

    raise InvalidSyntax(context, error["location"]["line"], error["message"])
