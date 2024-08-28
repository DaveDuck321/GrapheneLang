from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

from lark import Lark, Token, Tree, v_args
from lark.visitors import Interpreter


def is_list_of_str(
    items: Iterable[Any],
) -> TypeGuard[Iterable[str]]:
    # https://github.com/python/mypy/issues/3497#issuecomment-1083747764
    return all(isinstance(item, str) for item in items)


@dataclass
class ExpectedOutput:
    status: int
    stdout: list[str] | None
    stderr: list[str] | None


@dataclass
class TestConfig:
    temporarily_regressed: bool
    for_target: str | None
    compile_opts: ExpectedOutput | None
    compile_args: list[str]
    grep_ir_strs: list[str]
    run_opts: ExpectedOutput | None
    run_args: list[str]


# Global; only ever make one parser
lark = Lark.open(
    str(Path(__file__).parent / "test_config_grammar.lark"), parser="lalr", strict=True
)


class ConfigInterpreter(Interpreter):
    def __init__(self) -> None:
        super().__init__()

        self.config = TestConfig(False, None, None, [], [], None, [])

    @staticmethod
    def _format_msg(msg: str) -> list[str]:
        # No need to strip whitespace here, match_output() will do it later.
        return msg.strip("\n").split("\n")

    def _cmd_impl(
        self,
        status_tree: Tree | None,
        msg_tree: Tree | None,
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
    def regressed(self) -> None:
        self.config.temporarily_regressed = True

    @v_args(inline=True)
    def for_cmd(self, target: str) -> None:
        assert self.config.for_target is None

        self.config.for_target = target

    @v_args(inline=True)
    def compile_cmd(self, *trees: Tree) -> None:
        assert self.config.compile_opts is None

        *arg_tokens, status_tree, msg_tree = trees
        assert is_list_of_str(arg_tokens)

        self.config.compile_opts = self._cmd_impl(status_tree, msg_tree)
        self.config.compile_args.extend(arg_tokens)

    @v_args(inline=True)
    def run_cmd(self, *trees: Tree) -> None:
        assert self.config.run_opts is None

        *arg_tokens, status_tree, msg_tree = trees
        assert is_list_of_str(arg_tokens)

        self.config.run_opts = self._cmd_impl(status_tree, msg_tree)
        self.config.run_args.extend(arg_tokens)

    @v_args(inline=True)
    def grep_ir_cmd(self, token: Token) -> None:
        self.config.grep_ir_strs.append(token.strip())


def parse_file(path: Path) -> TestConfig:
    with path.open(encoding="utf-8") as file:
        lines = [
            line.removeprefix("///").strip() for line in file if line.startswith("///")
        ]

    tree = lark.parse("\n".join(lines))
    interpreter = ConfigInterpreter()
    interpreter.visit(tree)

    return interpreter.config
