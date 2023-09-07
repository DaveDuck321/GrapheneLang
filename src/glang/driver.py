import subprocess
import sys
from argparse import Action, ArgumentParser, Namespace
from collections.abc import Sequence
from os import getenv
from pathlib import Path
from typing import Any, Optional

from tap import Tap

from .graphene_parser import generate_ir_from_source
from .target import get_host_target, get_target, get_target_triple, load_target_config


class PrintHostTargetAction(Action):
    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        # Stop parsing and exit after printing the host's target. If we keep
        # parsing, then the parser complains that `input`, a required position
        # argument, is missing.
        print(get_host_target())
        parser.exit()


class DriverArguments(Tap):
    # TODO support multiple source files.
    input: Path
    output: Optional[Path] = None

    compile_to_object: bool = False
    """Do not run the linker, generate a target ".o" object file instead."""

    include_path: list[Path] = []
    """Add the specified directories to the search path for include files."""

    optimize: str = "0"
    """Optimization setting forwarded to clang."""

    emit_llvm: bool = False
    """Output generated IR to a file."""

    emit_llvm_to_stdout: bool = False
    """Output generated IR to stdout."""

    emit_optimized_llvm: bool = False
    """Output IR optimized by clang to a file."""

    emit_everything: bool = False
    """Output generated IR, optimized IR, and the binary."""

    debug_compiler: bool = False
    """Print full exception traces."""

    nostdlib: bool = False
    """Do not add the Graphene standard library to the include search path."""

    use_crt: bool = False
    """Link with the C runtime and standard library instead of Graphene's."""

    target: str = get_host_target()
    """Specify the architecture and platform to build for."""

    print_host_target: bool = False
    """Print the target corresponding to the host."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    def configure(self) -> None:
        self.add_argument("input")
        self.add_argument("-o", "--output")
        self.add_argument("-c", "--compile_to_object")
        self.add_argument("-I", "--include_path")
        self.add_argument("-O", "--optimize")
        self.add_argument("--print_host_target", nargs=0, action=PrintHostTargetAction)


def main() -> None:
    args = DriverArguments().parse_args()

    will_emit_llvm = args.emit_llvm or args.emit_everything
    will_emit_llvm_to_stdout = args.emit_llvm_to_stdout
    will_emit_optimized_llvm = args.emit_optimized_llvm or args.emit_everything
    will_emit_binary = (
        not (args.emit_llvm or args.emit_llvm_to_stdout) or args.emit_everything
    )

    llvm_output = args.input.with_suffix(".ll")
    optimized_llvm_output = llvm_output.with_suffix(".opt.ll")
    binary_output = Path("a.out")

    load_target_config(args.target)

    parent_dir = Path(__file__).parent.resolve()
    lib_dir = parent_dir / "lib"
    target_dir = lib_dir / "std" / get_target()

    if not args.nostdlib:
        args.include_path.append(lib_dir)
        args.include_path.append(target_dir)

    # -o defaults to binary output path
    if args.output:
        if will_emit_binary:
            binary_output = args.output
        else:
            llvm_output = args.output

    # Compile to ir
    ir = generate_ir_from_source(args.input, args.include_path, args.debug_compiler)

    if will_emit_llvm:
        with llvm_output.open("w", encoding="utf-8") as file:
            file.write(ir)

    if will_emit_llvm_to_stdout:
        sys.stdout.write(ir)

    # Use clang to finish compile
    assert args.optimize != "2", "@DaveDuck321's coward assert!"

    clang = getenv("GRAPHENE_CLANG_CMD", "clang")

    if will_emit_optimized_llvm:
        # TODO: the llvm optimization pipeline is run twice if we also want a
        # binary
        subprocess.run(
            [
                clang,
                "-S",
                "-emit-llvm",
                f"-O{args.optimize}",
                "-target",
                get_target_triple(),
                "-o",
                optimized_llvm_output,
                "-xir",  # Only STDIN is IR, so this should be last.
                "-",
            ],
            input=ir,
            text=True,
            encoding="utf-8",
            check=True,
        )

    extra_flags = []
    if args.compile_to_object:
        extra_flags.append("-c")
    if not args.use_crt and not args.compile_to_object:
        # -nostdlib prevents both the standard library and the start files from
        # being linked with the executable.
        extra_flags.append("-nostdlib")
        extra_flags.append(str(target_dir / "runtime.S"))

    if will_emit_binary:
        subprocess.run(
            [
                clang,
                f"-O{args.optimize}",
                "-fuse-ld=lld",  # Use the LLVM cross-linker.
                "-Wl,--build-id=sha1",
                "-static",
                "-target",
                get_target_triple(),
                "-o",
                binary_output,
                *extra_flags,
                "-xir",  # Only STDIN is IR, so this should be last.
                "-",
            ],
            input=ir,
            text=True,
            encoding="utf-8",
            check=True,
        )


if __name__ == "__main__":
    main()
