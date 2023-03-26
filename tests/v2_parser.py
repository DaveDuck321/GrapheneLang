from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from lark import Lark, Tree, v_args
from lark.visitors import Interpreter


@dataclass
class ExpectedOutput:
    status: int
    stdout: Optional[list[str]]
    stderr: Optional[list[str]]


@dataclass
class TestConfig:
    compile: Optional[ExpectedOutput]
    run: Optional[ExpectedOutput]


# Global; only ever make one parser
lark = Lark.open(str(Path(__file__).parent / "v2_grammar.lark"), parser="lalr")


class ConfigInterpreter(Interpreter):
    def __init__(self) -> None:
        super().__init__()

        self.config = TestConfig(None, None)

    @staticmethod
    def _format_msg(msg: str) -> list[str]:
        # No need to strip whitespace here, match_output() will do it later.
        return msg.strip("\n").split("\n")

    def _cmd_impl(
        self,
        status_tree: Optional[Tree],
        msg_tree: Optional[Tree],
    ) -> ExpectedOutput:
        expected_output = ExpectedOutput(0, None, None)

        if status_tree:
            (status,) = status_tree.children
            assert isinstance(status, str)

            expected_output.status = int(status)

        if msg_tree:
            stream, msg = msg_tree.children
            assert stream in ("OUT", "ERR")
            assert isinstance(msg, str)

            msg = self._format_msg(msg)

            if stream == "OUT":
                expected_output.stdout = msg
            else:
                expected_output.stderr = msg

        if expected_output.stderr and not status_tree:
            expected_output.status = 1

        return expected_output

    @v_args(inline=True)
    def compile_cmd(self, *trees: Tree) -> None:
        assert self.config.compile is None
        self.config.compile = self._cmd_impl(*trees)

    @v_args(inline=True)
    def run_cmd(self, *trees: Tree) -> None:
        assert self.config.run is None
        self.config.run = self._cmd_impl(*trees)


def parse_file(path: Path) -> TestConfig:
    with path.open(encoding="utf-8") as file:
        lines = map(
            lambda line: line.removeprefix("///").strip(),
            filter(lambda line: line.startswith("///"), file.readlines()),
        )

    tree = lark.parse("\n".join(lines))
    interpreter = ConfigInterpreter()
    interpreter.visit(tree)

    return interpreter.config
