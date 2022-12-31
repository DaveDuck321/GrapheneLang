import fnmatch
import json
import subprocess
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from pathlib import Path
from subprocess import PIPE
from threading import Lock
from typing import Optional

import schema

PARENT_DIR = Path(__file__).parent


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
    expected_output: dict[str, list[str]],
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

    def match_output(actual: list[str], expected: Optional[list[str]]) -> bool:
        if expected is None:
            return True

        assert isinstance(actual, list)
        assert isinstance(expected, list)

        if len(actual) != len(expected):
            return False

        actual_trimmed = map(str.strip, actual)
        expected_trimmed = map(str.strip, expected)

        # fnmatchcase() allows common Unix shell-style wildcards in expected
        # output, including:
        # - * to match everything
        # - ? to match any single character
        # - [seq] to match any character in seq
        # - [!seq] to match any character not in seq
        return all(map(fnmatch.fnmatchcase, actual_trimmed, expected_trimmed))

    if expected_output.get("status", 0) != status:
        raise TestFailure("status", status, stdout, stderr)

    if not match_output(stdout, expected_output.get("stdout")):
        raise TestFailure("stdout", status, stdout, stderr)

    if not match_output(stderr, expected_output.get("stderr")):
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


# Mutex to ensure prints remain ordered (within each test)
io_lock = Lock()


def run_test_print_result(test_dir: Path) -> bool:
    assert test_dir.is_dir()
    test_name = str(test_dir.relative_to(PARENT_DIR))

    try:
        run_test(test_dir)
        with io_lock:
            print(f"PASSED '{test_name}'")
        return True
    except TestFailure as error:
        with io_lock:
            print(f"FAILED '{test_name}'")
            print(error)
            print()
        return False


def run_tests(tests: list[Path], workers: int) -> int:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        passed = sum(executor.map(run_test_print_result, tests))

    failed = len(tests) - passed
    if failed:
        print(f"FAILED {failed}/{len(tests)} TESTS!")
    else:
        print(f"PASSED ALL {passed} TESTS")

    return failed


def main() -> None:
    parser = ArgumentParser("run_tests.py")
    parser.add_argument("--test", required=False)
    parser.add_argument("--workers", required=False, type=int, default=cpu_count())

    args = parser.parse_args()
    if args.test is not None:
        run_test(PARENT_DIR / args.test, io_harness=False)
    else:
        all_test_dirs = [
            test_path.parent
            for test_path in PARENT_DIR.rglob("**/test.json")
            if test_path.is_file()
        ]

        exit(run_tests(all_test_dirs, args.workers))


if __name__ == "__main__":
    main()
