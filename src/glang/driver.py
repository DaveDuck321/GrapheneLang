from __future__ import annotations

import subprocess
import sys
from argparse import Action, ArgumentParser, Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from os import fdopen, getenv
from pathlib import Path
from tempfile import TemporaryDirectory, mkstemp
from typing import Any

from tap import Tap

from glang.graphene_parser import generate_ir_from_source
from glang.target import (
    get_host_target,
    get_target,
    get_target_triple,
    load_target_config,
)

global_tmp_dir = TemporaryDirectory()


def run_checked(args: list[str | Path], stdin: bytes | None = None) -> bytes:
    # Like the python one but actually output the stderr on fail
    result = subprocess.run(
        args,
        input=stdin,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        command_str = " ".join(str(arg) for arg in args)
        print(f"Error in command: {command_str}", file=sys.stderr)
        print(result.stderr.decode("utf-8"), file=sys.stderr)
        exit(1)

    return result.stdout


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
    inputs: list[Path]
    output: Path | None = None

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
        self.add_argument("inputs")
        self.add_argument("-o", "--output")
        self.add_argument("-c", "--compile_to_object")
        self.add_argument("-I", "--include_path")
        self.add_argument("-O", "--optimize")
        self.add_argument("--print_host_target", nargs=0, action=PrintHostTargetAction)


@dataclass(frozen=True)
class Stage:
    source: Path | bytes
    derived_from: Stage | None

    def get_bytes(self) -> bytes:
        if isinstance(self.source, bytes):
            return self.source

        assert isinstance(self.source, Path)
        return self.source.read_bytes()

    def get_file(self) -> Path:
        if isinstance(self.source, Path):
            return self.source

        fd, path = mkstemp(dir=Path(global_tmp_dir.name))
        with fdopen(fd, "wb") as file:
            file.write(self.source)

        return Path(path)

    def get_top_level_source_file(self) -> Path:
        if self.derived_from is None:
            return self.get_file()
        return self.derived_from.get_top_level_source_file()


class ELF_Binary(Stage):
    pass


class Assembly(Stage):
    def compile(self, args: DriverArguments) -> ELF_Binary:
        expanded = run_checked(
            [
                getenv("GRAPHENE_CLANG_CMD", "clang"),
                "-E",
                "-xassembler-with-cpp",
                self.get_file(),
            ],
        )
        compiled = run_checked(
            [
                getenv("GRAPHENE_LLVM_MC_CMD", "llvm-mc"),
                "--filetype=obj",
                f"--triple={get_target_triple()}",
                "-g",
            ],
            stdin=expanded,
        )
        return ELF_Binary(compiled, self)


class LLVM_IR(Stage):
    def compile(self, args: DriverArguments) -> ELF_Binary:
        result = run_checked(
            [
                getenv("GRAPHENE_LLC_CMD", "llc"),
                "--relocation-model=static",
                f"--mtriple={get_target_triple()}",
                "--filetype=obj",
                f"-O{args.optimize}",
                "-",
            ],
            stdin=self.get_bytes(),
        )
        return ELF_Binary(result, self)

    def optimize(self, args: DriverArguments) -> LLVM_IR:
        will_emit_optimized_llvm = args.emit_optimized_llvm or args.emit_everything
        optimized_ir = run_checked(
            [
                getenv("GRAPHENE_OPT_CMD", "opt"),
                "-S",
                f"-O{args.optimize}",
                "-",
            ],
            stdin=self.get_bytes(),
        )

        if will_emit_optimized_llvm:
            if args.emit_everything or args.output is None:
                optimized_llvm_output = self.get_top_level_source_file().with_suffix(
                    ".opt.ll"
                )
            else:
                optimized_llvm_output = args.output
            optimized_llvm_output.write_bytes(optimized_ir)

        return LLVM_IR(optimized_ir, self)


class GrapheneSource(Stage):
    def compile(self, args: DriverArguments) -> LLVM_IR:
        will_emit_llvm = args.emit_llvm or args.emit_everything

        ir = generate_ir_from_source(
            self.get_file(), args.include_path, debug_compiler=args.debug_compiler
        )
        ir_bytes = ir.encode("utf-8")

        if args.emit_llvm_to_stdout:
            print(ir_bytes.decode("utf-8"))

        if will_emit_llvm:
            if args.emit_everything or args.output is None:
                llvm_output = self.get_top_level_source_file().with_suffix(".ll")
            else:
                llvm_output = args.output
            llvm_output.write_bytes(ir_bytes)

        return LLVM_IR(ir_bytes, self)


def link_and_output(args: DriverArguments, objects: list[ELF_Binary]) -> None:
    linker_args: list[str] = []

    if not args.use_crt:
        linker_args.extend(["--nostdlib", "--static"])

    binary_output = args.output or Path("a.out")
    run_checked(
        [
            getenv("GRAPHENE_LLD_CMD", "ld.lld"),
            "--build-id=sha1",
            *linker_args,
            *(obj.get_file() for obj in objects),
            "-o",
            binary_output,
        ],
    )


def bundle_and_output(args: DriverArguments, objects: list[ELF_Binary]) -> None:
    lib_output = args.output or Path("lib.a")
    run_checked(
        [
            getenv("GRAPHENE_LLVM_AR_CMD", "llvm-ar"),
            "rcs",
            lib_output,
            *(obj.get_file() for obj in objects),
        ],
    )


def main() -> None:
    args = DriverArguments().parse_args()
    will_emit_binary = (
        not (args.emit_llvm or args.emit_optimized_llvm or args.emit_llvm_to_stdout)
        or args.emit_everything
    )

    load_target_config(args.target)

    parent_dir = Path(__file__).parent.resolve()
    lib_dir = parent_dir / "lib"
    target_dir = lib_dir / "std" / get_target()

    if not args.nostdlib:
        args.include_path.append(lib_dir)
        args.include_path.append(target_dir)

    assert args.optimize != "2", "@DaveDuck321's coward assert!"

    # 1) Graphene frontend
    sources: list[GrapheneSource] = [
        GrapheneSource(source, None) for source in args.inputs
    ]
    llvm_sources: list[LLVM_IR] = [source.compile(args) for source in sources]

    # 2) Optimize (if enabled)
    if args.optimize == "0":
        opt_sources = llvm_sources  # Don't even touch on O0
    else:
        opt_sources = [llvm.optimize(args) for llvm in llvm_sources]

    # 3) Compile
    if not will_emit_binary:
        return  # Already done

    objects = [source.compile(args) for source in opt_sources]
    if not args.use_crt and not args.compile_to_object:
        # We're not using libc, compile our own runtime
        asm = Assembly(target_dir / "runtime.S", None)
        objects.append(asm.compile(args))

    # 4) Create output objects if requested
    if args.compile_to_object:
        if len(objects) > 1:
            # Static library
            bundle_and_output(args, objects)
        else:
            # Object
            target_output = objects[0].get_top_level_source_file().with_suffix(".o")
            target_output.write_bytes(objects[0].get_bytes())
        return

    # 5) Link
    link_and_output(args, objects)


if __name__ == "__main__":
    main()
