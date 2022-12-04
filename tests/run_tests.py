import json
import schema
import subprocess

from argparse import ArgumentParser
from pathlib import Path
from subprocess import PIPE

all_tests = [
    "c_demo/compile_error",
    "c_demo/runtime_status",
    "order_of_operation",
    "subexpression_ordering",
    "undeclared_variable",
    "uninitialized_variable_usage",
    "string_constants",
    "string_constants_ir",
]


class TestFailure(RuntimeError):
    def __init__(self, name, status, stdout, stderr, stage=None):
        super().__init__(f"Actual '{name}' does not match expected")

        self.name = name
        self.status = status
        self.stdout = stdout
        self.stderr = stderr
        self.stage = stage

    def __str__(self) -> str:
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


def validate_command_status(directory: Path, command: list[str], expected_output):
    status = subprocess.call(command, cwd=str(directory))

    if expected_output.get("status", 0) != status:
        print("*** ERROR: Actual 'status' does not match expected")


def validate_command_output_with_harness(
    directory: Path,
    command: list[str],
    expected_output,
):

    result = subprocess.run(
        command, cwd=str(directory), stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True
    )
    status = result.returncode
    stdout, stderr = result.stdout.splitlines(), result.stderr.splitlines()

    if expected_output.get("status", 0) != status:
        raise TestFailure("status", status, stdout, stderr)

    if expected_output.get("stdout", stdout) != stdout:
        raise TestFailure(f"stdout", status, stdout, stderr)

    if expected_output.get("stderr", stderr) != stderr:
        raise TestFailure(f"stderr", status, stdout, stderr)

    return True


def run_test(path: Path, io_harness=True):
    # Load/ validate the test
    config_path = path / "test.json"
    assert config_path.exists()

    (path / "out").mkdir(exist_ok=True)

    config = json.load(config_path.open())
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
    except TestFailure as e:
        raise TestFailure(e.name, e.status, e.stdout, e.stderr, stage="compile")

    if "runtime" not in config:
        return

    try:
        fn_validate(
            path,
            config["runtime"].get("command", []),
            config["runtime"].get("output", {}),
        )
    except TestFailure as e:
        raise TestFailure(e.name, e.status, e.stdout, e.stderr, stage="runtime")


def run_tests(tests):
    passed = []
    failed = []

    for i, test in enumerate(tests):
        print(f"TEST {i+1}: '{test}'")
        try:
            run_test(Path(__file__).parent / test)
            passed.append(test)
            print("   PASSED")
        except TestFailure as error:
            failed.append(test)
            print(error)
            print()

    if len(failed) == 0:
        print(f"PASSED {len(passed)} TESTS")
        return 0

    print(f"FAILED {len(failed)}/{len(tests)} TESTS!!")
    return len(failed)


if __name__ == "__main__":
    parser = ArgumentParser("run_tests.py")
    parser.add_argument("--test", required=False)

    args = parser.parse_args()
    if args.test is not None:
        run_test(Path(__file__).parent / args.test, io_harness=False)
    else:
        exit(run_tests(all_tests))
