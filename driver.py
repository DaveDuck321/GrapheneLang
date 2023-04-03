import argparse
import subprocess
import sys
from os import getenv
from parser import generate_ir_from_source
from pathlib import Path


def extract_include_paths(args: list[str]) -> tuple[list[Path], list[str]]:
    include_path = []
    filtered_args = []
    for arg in args:
        if arg.startswith("-I"):
            path = Path(arg[2:])
            include_path.append(path)
        else:
            filtered_args.append(arg)

    return include_path, filtered_args


def extract_optimization_level(args: list[str]) -> tuple[str, list[str]]:
    level = "0"  # Default
    filtered_args = []
    for arg in args:
        if arg.startswith("-O"):
            level = arg[2:]
        else:
            filtered_args.append(arg)

    assert level != "2"  # Coward assert
    return level, filtered_args


def main() -> None:
    include_path, sys_args = extract_include_paths(sys.argv[1:])
    opt_level, sys_args = extract_optimization_level(sys_args)

    parser = argparse.ArgumentParser("python driver.py")
    # TODO: support multiple source files
    parser.add_argument("input", nargs=1, type=Path)
    parser.add_argument("-o", "--output", required=False, type=Path)
    parser.add_argument("-c", "--compile-to-object", action="store_true")
    parser.add_argument("-I<include path>", action="store_true")  # Dummy arg
    parser.add_argument("-O<level>", action="store_true")  # Dummy arg
    parser.add_argument("--emit-llvm", action="store_true")
    parser.add_argument("--emit-optimized-llvm", action="store_true")
    parser.add_argument("--emit-everything", action="store_true")
    parser.add_argument("--debug-compiler", action="store_true")
    parser.add_argument("--nostdlib", action="store_true")
    parser.add_argument("--use-crt", action="store_true")
    args = parser.parse_args(sys_args)

    will_emit_llvm: bool = args.emit_llvm or args.emit_everything
    will_emit_optimized_llvm: bool = args.emit_optimized_llvm or args.emit_everything
    will_emit_binary: bool = not args.emit_llvm or args.emit_everything

    llvm_output: Path = args.input[0].with_suffix(".ll")
    optimized_llvm_output: Path = llvm_output.with_suffix(".opt.ll")
    binary_output = Path("a.out")

    parent_dir = Path(__file__).parent.resolve()

    if not args.nostdlib:
        include_path.append(parent_dir)

    # -o defaults to binary output path
    if args.output:
        if will_emit_binary:
            binary_output = args.output
        else:
            llvm_output = args.output

    # Compile to ir
    ir = generate_ir_from_source(args.input[0], include_path, args.debug_compiler)

    if will_emit_llvm:
        with llvm_output.open("w", encoding="utf-8") as file:
            file.write(ir)

    # Use clang to finish compile

    if will_emit_optimized_llvm:
        # TODO: the llvm optimization pipeline is run twice if we also want a binary
        subprocess.run(
            [
                getenv("GRAPHENE_CLANG_CMD", "clang"),
                "-Wno-override-module",
                "-S",
                "-emit-llvm",
                f"-O{opt_level}",
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
        extra_flags.append("-nostartfiles")
        extra_flags.append("-nostdlib")
        extra_flags.append(str(parent_dir / "std" / "runtime.S"))

    if will_emit_binary:
        subprocess.run(
            [
                getenv("GRAPHENE_CLANG_CMD", "clang"),
                "-Wno-override-module",  # Don't complain about the target triple.
                f"-O{opt_level}",
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
