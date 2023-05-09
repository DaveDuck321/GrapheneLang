import fnmatch
import subprocess
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from os import getenv
from pathlib import Path
from sys import exit as sys_exit
from threading import Lock
from typing import Optional

from test_config_parser import ExpectedOutput, parse_file

LLI_CMD = getenv("GRAPHENE_LLI_CMD", "lli")

PARENT_DIR = Path(__file__).parent
TESTS_DIR = PARENT_DIR
OUT_DIR = TESTS_DIR / "out"
RUNTIME_OBJ_PATH = OUT_DIR / "runtime.o"
DRIVER_PATH = PARENT_DIR.parent / "driver.py"


class TestFailure(RuntimeError):
    pass


class TestCommandFailure(TestFailure):
    def __init__(
        self, name: str, status: int, stdout: list[str], stderr: list[str], stage: str
    ) -> None:
        super().__init__(f"Actual '{name}' does not match expected")

        self.name = name
        self.status = status
        self.stdout = stdout
        self.stderr = stderr
        self.stage = stage

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
    stage: str,
    directory: Path,
    command: list[str | Path],
    expected_output: ExpectedOutput,
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

    if expected_output.status != status:
        raise TestCommandFailure("status", status, stdout, stderr, stage)

    if not match_output(stdout, expected_output.stdout):
        raise TestCommandFailure("stdout", status, stdout, stderr, stage)

    if not match_output(stderr, expected_output.stderr):
        raise TestCommandFailure("stderr", status, stdout, stderr, stage)

    # Return the unsplit output.
    return result.stdout


def run_test(file_path: Path) -> None:
    assert file_path.exists()
    config = parse_file(file_path)

    # @COMPILE (mandatory)
    assert config.compile
    ir_output = run_command(
        "compile",
        file_path.parent,
        [
            "python",
            DRIVER_PATH,
            "--emit-llvm-to-stdout",
            *config.compile_args,
            file_path,
        ],
        config.compile,
    )

    # @GREP_IR
    if config.grep_ir_str and config.grep_ir_str not in ir_output:
        raise IRGrepFailure(config.grep_ir_str)

    # @RUN
    if config.run:
        # If we're not using the C runtime, then don't resolve lli process
        # symbols in JIT'd code (or else the tests will be able to call into
        # glibc) and specify `_start` as the entry function (defined in
        # runtime.S).
        # NOTE lli has a hardcoded runtime initialisation routine, which
        # includes running constructors and destructors. This always gets called
        # before execution jumps into the JIT'd code.
        lli_runtime_options = (
            ["--no-process-syms", "--entry-function=_start"]
            if "--use-crt" not in config.compile_args
            else []
        )

        run_command(
            "runtime",
            TESTS_DIR,
            [
                LLI_CMD,
                *lli_runtime_options,
                "--extra-object",
                RUNTIME_OBJ_PATH,
                "-",
                *config.run_args,
            ],
            config.run,
            ir_output,
        )


# Mutex to ensure prints remain ordered (within each test)
io_lock = Lock()


def run_test_print_result(test_path: Path) -> bool:
    test_name = str(test_path.relative_to(PARENT_DIR))

    try:
        run_test(test_path)

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
            ["cc", "-c", runtime_src_path, "-o", RUNTIME_OBJ_PATH], check=True
        )

    assert RUNTIME_OBJ_PATH.is_file()


def main() -> None:
    parser = ArgumentParser("run_tests.py")
    parser.add_argument("--test", required=False)
    parser.add_argument("--workers", required=False, type=int, default=cpu_count())

    args = parser.parse_args()

    assert DRIVER_PATH.is_file()
    assert TESTS_DIR.is_dir()
    OUT_DIR.mkdir(exist_ok=True)

    build_jit_dependencies()

    if args.test is not None:
        test_path = PARENT_DIR / args.test

        run_test_print_result(test_path)
    else:
        all_test_files = [
            test_file
            for test_file in TESTS_DIR.rglob("**/*.c3")
            if not test_file.stem.startswith("_") and test_file.is_file()
        ]

        sys_exit(run_tests(all_test_files, args.workers))


if __name__ == "__main__":
    main()
