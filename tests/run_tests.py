import subprocess
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from fnmatch import fnmatchcase
from importlib import resources
from multiprocessing import cpu_count
from os import getenv
from pathlib import Path
from sys import exit as sys_exit
from threading import Lock

from tap import Tap
from test_config_parser import ExpectedOutput, TestConfig, parse_file

LLI_CMD = getenv("GRAPHENE_LLI_CMD", "lli")
OPT_CMD = getenv("GRAPHENE_OPT_CMD", "opt")

PARENT_DIR = Path(__file__).parent
TESTS_DIR = PARENT_DIR
OUT_DIR = TESTS_DIR / "out"
RUNTIME_OBJ_PATH = OUT_DIR / "runtime.o"
DRIVER = "glang"

HOST_TARGET = subprocess.run(
    [DRIVER, "--print-host-target"],
    check=True,
    stdout=subprocess.PIPE,
    text=True,
).stdout.strip()


class TestStatus(Enum):
    PASSED = 0
    FAILED = 1
    SKIPPED_DUE_TO_TARGET = 2
    EXPECTED_FAIL = 3
    UNEXPECTED_PASS = 4


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
    stdin: str | None = None,
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

    def match_output(actual: list[str], expected: list[str] | None) -> bool:
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
        return all(map(fnmatchcase, actual_trimmed, expected_trimmed))

    if expected_output.status != status:
        raise TestCommandFailure("status", status, stdout, stderr, stage)

    if not match_output(stdout, expected_output.stdout):
        raise TestCommandFailure("stdout", status, stdout, stderr, stage)

    if not match_output(stderr, expected_output.stderr):
        raise TestCommandFailure("stderr", status, stdout, stderr, stage)

    # Return the unsplit output.
    return result.stdout


def run_test_throw_on_fail(config: TestConfig, file_path: Path) -> None:
    # @COMPILE (mandatory)
    assert config.compile_opts
    ir_output = run_command(
        "compile",
        file_path.parent,
        [
            DRIVER,
            "--emit-llvm-to-stdout",
            *config.compile_args,
            file_path,
        ],
        config.compile_opts,
    )

    # @OPTIMIZE
    if config.optimize_args is not None:
        ir_output = run_command(
            "optimize",
            file_path.parent,
            [
                OPT_CMD,
                "-S",
                "-O3",
                *config.optimize_args,
                "-",
            ],
            expected_output=ExpectedOutput(0, None, None),
            stdin=ir_output,
        )

    # @GREP_IR
    for needle in config.grep_ir_strs:
        if not fnmatchcase(ir_output, f"*{needle}*"):
            raise IRGrepFailure(needle)

    # @RUN
    if config.run_opts:
        # If we're not using the C runtime, then don't resolve lli process
        # symbols in JIT'd code (or else the tests will be able to call into
        # glibc) and specify `_start` as the entry function (defined in
        # runtime.S).
        # NOTE lli has a hardcoded runtime initialisation routine, which
        # includes running constructors and destructors. This always gets called
        # before execution jumps into the JIT'd code.
        lli_runtime_options = (
            ["--no-process-syms", "--entry-function=_lli_start"]
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
                "--lljit-platform=Inactive",
                "-",
                *config.run_args,
            ],
            config.run_opts,
            ir_output,
        )


# Mutex to ensure prints remain ordered (within each test)
io_lock = Lock()


def run_test_print_result(test_path: Path) -> TestStatus:
    assert test_path.exists()
    config = parse_file(test_path)

    test_name = str(test_path.relative_to(PARENT_DIR))

    if config.for_target is not None and config.for_target != HOST_TARGET:
        # Skip this test.
        with io_lock:
            print(f"SKIPPED '{test_name}'")
        return TestStatus.SKIPPED_DUE_TO_TARGET

    try:
        run_test_throw_on_fail(config, test_path)

        # Test has passed!
        if config.expected_failing:
            with io_lock:
                print(f"UNEXPECTED PASS '{test_name}'")
            return TestStatus.UNEXPECTED_PASS
        else:
            with io_lock:
                print(f"PASSED '{test_name}'")
            return TestStatus.PASSED

    except TestFailure as error:
        if config.expected_failing:
            with io_lock:
                print(f"FAILED '{test_name}' (but marked expected)")
            return TestStatus.EXPECTED_FAIL
        else:
            with io_lock:
                print(f"FAILED '{test_name}'")
                print(error)
                print()

            return TestStatus.FAILED


def run_tests(tests: list[Path], workers: int) -> int:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        try:
            results = list(executor.map(run_test_print_result, tests))
        except KeyboardInterrupt:
            # Stop the currently running tests from spamming the output.
            io_lock.acquire()
            return 0

    def count_status(status: TestStatus) -> int:
        return sum(1 for result in results if result == status)

    passed = count_status(TestStatus.PASSED)
    failed = count_status(TestStatus.FAILED)
    expected_fails = count_status(TestStatus.EXPECTED_FAIL)
    unexpected_passes = count_status(TestStatus.UNEXPECTED_PASS)
    skipped_target = count_status(TestStatus.SKIPPED_DUE_TO_TARGET)

    tests_that_would_ideally_pass = passed + failed + expected_fails + unexpected_passes

    if failed + unexpected_passes:
        print(f"FAILED {failed}/{tests_that_would_ideally_pass} TESTS!")

        if unexpected_passes:
            print(f"INCORRECTLY PASSED {unexpected_passes} TEST MARKED AS FAILURE")

        return 1  # Exit failure
    else:
        if skipped_target:
            print(f"SKIPPED {skipped_target} TESTS DUE TO INCOMPATIBLE TARGET")

        print(f"PASSED {passed}/{tests_that_would_ideally_pass} TESTS")

        if expected_fails:
            print(
                f"WARNING: {expected_fails}/{tests_that_would_ideally_pass} TESTS ARE MARKED AS EXPECTED FAILURES"
            )

    return 0  # Exit success


def build_jit_dependencies() -> None:
    with resources.as_file(
        resources.files("glang.lib.std") / HOST_TARGET / "runtime.S"
    ) as runtime_src_path:
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


class Arguments(Tap):
    test: Path | None = None
    workers: int = cpu_count()


def main() -> None:
    args = Arguments().parse_args()

    assert TESTS_DIR.is_dir()
    OUT_DIR.mkdir(exist_ok=True)

    build_jit_dependencies()

    if args.test is not None:
        test_path = args.test if args.test.is_file() else PARENT_DIR / args.test
        try:
            run_test_print_result(test_path.resolve(strict=True))
        except FileNotFoundError:
            print(f"No such file: '{test_path}'")
    else:
        all_test_files = [
            test_file
            for test_file in TESTS_DIR.rglob("**/*.c3")
            if not test_file.stem.startswith("_") and test_file.is_file()
        ]

        sys_exit(run_tests(all_test_files, args.workers))


if __name__ == "__main__":
    main()
