import subprocess
import sys
from os import getenv
from pathlib import Path
from typing import Optional

from tap import Tap

from graphene_parser import generate_ir_from_source
from target import get_target, get_target_triple, load_target_config


class DriverArguments(Tap):
    # TODO support multiple source files.
    # TODO target the host by default.
    input: Path
    output: Optional[Path] = None
    compile_to_object: bool = False
    include_path: list[Path] = []
    optimize: str = "0"
    emit_llvm: bool = False
    emit_llvm_to_stdout: bool = False
    emit_optimized_llvm: bool = False
    emit_everything: bool = False
    debug_compiler: bool = False
    nostdlib: bool = False
    use_crt: bool = False
    target: str = "x86_64_linux"
    static: bool = False

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    def configure(self) -> None:
        self.add_argument("input")
        self.add_argument("-o", "--output")
        self.add_argument("-c", "--compile_to_object")
        self.add_argument("-I", "--include_path")
        self.add_argument("-O", "--optimize")


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

    parent_dir = Path(__file__).parent.resolve()

    if not args.nostdlib:
        args.include_path.append(parent_dir)

    # -o defaults to binary output path
    if args.output:
        if will_emit_binary:
            binary_output = args.output
        else:
            llvm_output = args.output

    # Compile to ir
    load_target_config(args.target)
    ir = generate_ir_from_source(args.input, args.include_path, args.debug_compiler)

    if will_emit_llvm:
        with llvm_output.open("w", encoding="utf-8") as file:
            file.write(ir)

    if will_emit_llvm_to_stdout:
        sys.stdout.write(ir)

    # Use clang to finish compile

    if will_emit_optimized_llvm:
        # TODO: the llvm optimization pipeline is run twice if we also want a
        # binary
        subprocess.run(
            [
                getenv("GRAPHENE_CLANG_CMD", "clang"),
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
        extra_flags.append(str(parent_dir / "std" / get_target() / "runtime.S"))
    if args.static:
        extra_flags.append("-static")

    if will_emit_binary:
        subprocess.run(
            [
                getenv("GRAPHENE_CLANG_CMD", "clang"),
                f"-O{args.optimize}",
                "-fuse-ld=lld",  # Use the LLVM cross-linker.
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
