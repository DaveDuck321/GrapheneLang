import fnmatch
import json
import subprocess
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from multiprocessing import cpu_count
from os import getenv
from pathlib import Path
from sys import exit as sys_exit
from threading import Lock
from typing import Optional

import schema
from v2_parser import parse_file

LLI_CMD = getenv("GRAPHENE_LLI_CMD", "lli")

PARENT_DIR = Path(__file__).parent
V2_TESTS_DIR = PARENT_DIR / "v2"
V2_OUT_DIR = V2_TESTS_DIR / "out"
RUNTIME_OBJ_PATH = V2_OUT_DIR / "runtime.o"
DRIVER_PATH = PARENT_DIR.parent / "driver.py"


class TestFailure(RuntimeError):
    pass


class TestCommandFailure(TestFailure):
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

    def with_stage(self, stage: str) -> "TestCommandFailure":
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


class IRGrepFailure(TestFailure):
    def __init__(self, needle: str) -> None:
        super().__init__(f"Couldn't find `{needle}` in the IR")

        self.needle = needle

    def __str__(self) -> str:
        return f"***GREP_IR ERROR: {super().__str__()}"


def run_command(
    directory: Path,
    command: list[str | Path],
    expected_output: dict[str, list[str]],
    stdin: Optional[str] = None,
) -> str:
    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
        cwd=str(directory),
        input=stdin,
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
        raise TestCommandFailure("status", status, stdout, stderr)

    if not match_output(stdout, expected_output.get("stdout")):
        raise TestCommandFailure("stdout", status, stdout, stderr)

    if not match_output(stderr, expected_output.get("stderr")):
        raise TestCommandFailure("stderr", status, stdout, stderr)

    # Return the unsplit output.
    return result.stdout


def run_v1_test(path: Path) -> None:
    # Load/ validate the test
    config_path = path / "test.json"
    assert config_path.exists()

    (path / "out").mkdir(exist_ok=True)

    config: dict[str, dict] = json.load(config_path.open())
    schema.validate_config_follows_schema(config)

    # Run the test
    try:
        run_command(
            path,
            config["compile"].get("command", []),
            config["compile"].get("output", {}),
        )
    except TestCommandFailure as exc:
        raise exc.with_stage("compile")

    if "runtime" not in config:
        return

    try:
        run_command(
            path,
            config["runtime"].get("command", []),
            config["runtime"].get("output", {}),
        )
    except TestCommandFailure as exc:
        raise exc.with_stage("runtime")


def run_v2_test(file_path: Path) -> None:
    assert file_path.exists()
    config = parse_file(file_path)

    # Compile the test
    assert config.compile
    try:
        ir_output = run_command(
            file_path.parent,
            [
                "python",
                DRIVER_PATH,
                "--emit-llvm-to-stdout",
                *config.compile_args,
                file_path,
            ],
            asdict(config.compile),
        )
    except TestCommandFailure as exc:
        raise exc.with_stage("compile")

    if config.grep_ir_str:
        if config.grep_ir_str not in ir_output:
            raise IRGrepFailure(config.grep_ir_str)

    if config.run is None:
        return

    try:
        run_command(
            V2_TESTS_DIR,
            [LLI_CMD, "--extra-object", RUNTIME_OBJ_PATH, "-"],
            asdict(config.run),
            ir_output,
        )
    except TestCommandFailure as exc:
        raise exc.with_stage("runtime")


def is_v2_test(path: Path) -> bool:
    return not path.is_dir()


# Mutex to ensure prints remain ordered (within each test)
io_lock = Lock()


def run_test_print_result(test_path: Path) -> bool:
    test_name = str(test_path.relative_to(PARENT_DIR))

    try:
        if is_v2_test(test_path):
            run_v2_test(test_path)
        else:
            run_v1_test(test_path)

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


def build_jit_dependencies() -> None:
    runtime_src_path = PARENT_DIR.parent / "std" / "runtime.S"
    assert runtime_src_path.is_file()

    # Don't compile the runtime again if it's up-to-date.
    if (
        not RUNTIME_OBJ_PATH.is_file()
        or runtime_src_path.stat().st_mtime > RUNTIME_OBJ_PATH.stat().st_mtime
    ):
        # Need to use `cc` because `as` doesn't run the C preprocessor.
        subprocess.run(
            ["cc", "-c", str(runtime_src_path), "-o", str(RUNTIME_OBJ_PATH)], check=True
        )

    assert RUNTIME_OBJ_PATH.is_file()


def main() -> None:
    parser = ArgumentParser("run_tests.py")
    parser.add_argument("--test", required=False)
    parser.add_argument("--workers", required=False, type=int, default=cpu_count())

    args = parser.parse_args()

    assert DRIVER_PATH.is_file()
    assert V2_TESTS_DIR.is_dir()
    V2_OUT_DIR.mkdir(exist_ok=True)

    build_jit_dependencies()

    if args.test is not None:
        test_path = PARENT_DIR / args.test

        run_test_print_result(test_path)
    else:
        all_v1_test_dirs = [
            test_path.parent
            for test_path in PARENT_DIR.rglob("**/test.json")
            if test_path.is_file()
        ]
        all_v2_test_files = [
            test_file
            for test_file in V2_TESTS_DIR.rglob("**/*.c3")
            if not test_file.stem.startswith("_") and test_file.is_file()
        ]

        sys_exit(run_tests(all_v1_test_dirs + all_v2_test_files, args.workers))


if __name__ == "__main__":
    main()
