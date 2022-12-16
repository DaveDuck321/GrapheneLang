import json
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from subprocess import PIPE
from typing import Optional

import schema

all_tests = [
    "c_demo/compile_error",
    "c_demo/runtime_status",
    "duplicate_generics",
    "generics_parsing",
    "order_of_operation",
    "simple_typedef",
    "string_constants",
    "string_constants_ir",
    "subexpression_ordering",
    "unary_operators",
    "undeclared_variable",
    "uninitialized_variable_usage",
]


class TestFailure(RuntimeError):
    def __init__(
        self,
        name: str,
        status: int,
        stdout: list[str],
        stderr: list[str],
    ) -> None:
        super().__init__(f"Actual '{name}' does not match expected")

        self.name = name
        self.status = status
        self.stdout = stdout
        self.stderr = stderr

        self.stage: Optional[str] = None

    def with_stage(self, stage: str) -> "TestFailure":
        self.stage = stage

        return self

    def __str__(self) -> str:
        assert self.stage is not None

        return "\n".join(
            [
                f"***{self.stage.upper()} ERROR: Actual '{self.name}' does not match expected",
                "====status====",
                str(self.status),
                "====stdout====",
                *self.stdout,
                "====stderr====",
                *self.stderr,
            ]
        )


def validate_command_status(
    directory: Path, command: list[str], expected_output
) -> None:
    status = subprocess.call(command, cwd=str(directory))

    if expected_output.get("status", 0) != status:
        print("*** ERROR: Actual 'status' does not match expected")


def validate_command_output_with_harness(
    directory: Path,
    command: list[str],
    expected_output,
):

    result = subprocess.run(
        command,
        check=False,
        cwd=str(directory),
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        text=True,
    )
    status = result.returncode
    stdout, stderr = result.stdout.splitlines(), result.stderr.splitlines()

    if expected_output.get("status", 0) != status:
        raise TestFailure("status", status, stdout, stderr)

    if expected_output.get("stdout", stdout) != stdout:
        raise TestFailure("stdout", status, stdout, stderr)

    if expected_output.get("stderr", stderr) != stderr:
        raise TestFailure("stderr", status, stdout, stderr)

    return True


def run_test(path: Path, io_harness=True) -> None:
    # Load/ validate the test
    config_path = path / "test.json"
    assert config_path.exists()

    (path / "out").mkdir(exist_ok=True)

    config: dict[str, dict] = json.load(config_path.open())
    schema.validate_config_follows_schema(config)

    fn_validate = (
        validate_command_output_with_harness if io_harness else validate_command_status
    )

    # Run the test
    try:
        fn_validate(
            path,
            config["compile"].get("command", []),
            config["compile"].get("output", {}),
        )
    except TestFailure as exc:
        raise exc.with_stage("compile")

    if "runtime" not in config:
        return

    try:
        fn_validate(
            path,
            config["runtime"].get("command", []),
            config["runtime"].get("output", {}),
        )
    except TestFailure as exc:
        raise exc.with_stage("runtime")


def run_tests(tests: list[str]) -> int:
    passed: int = 0
    failed: int = 0

    for i, test in enumerate(tests, 1):
        print(f"TEST {i}: '{test}'")
        try:
            run_test(Path(__file__).parent / test)
            passed += 1
            print("   PASSED")
        except TestFailure as error:
            failed += 1
            print(error)
            print()

    if failed:
        print(f"FAILED {failed}/{len(tests)} TESTS!")
    else:
        print(f"PASSED {passed} TESTS")

    return failed


if __name__ == "__main__":
    parser = ArgumentParser("run_tests.py")
    parser.add_argument("--test", required=False)

    args = parser.parse_args()
    if args.test is not None:
        run_test(Path(__file__).parent / args.test, io_harness=False)
    else:
        exit(run_tests(all_tests))
