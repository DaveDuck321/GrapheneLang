from parser import generate_ir_from_source

from pathlib import Path
import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser("python driver.py")
    # TODO: support multiple source files
    parser.add_argument("input", nargs=1, type=Path)
    parser.add_argument("-o", "--output", required=False, type=Path)
    parser.add_argument("--emit-llvm", action="store_true")
    parser.add_argument("--emit-everything", action="store_true")
    args = parser.parse_args()

    will_emit_llvm = args.emit_llvm or args.emit_everything
    will_emit_binary = not args.emit_llvm or args.emit_everything

    llvm_output = args.input[0].with_suffix(".ll")
    binary_output = Path("a.out")

    # -o defaults to binary output path
    if args.output:
        if will_emit_binary:
            binary_output = args.output
        else:
            llvm_output = args.output

    # Compile to ir
    ir = generate_ir_from_source(args.input[0].open().read())

    if will_emit_llvm:
        llvm_output.open("w").write(ir)

    # Use clang to finish compile
    if will_emit_binary:
        if will_emit_llvm:
            subprocess.check_call(["clang", llvm_output, "-o", binary_output])
        else:
            subprocess.run(
                ["clang", "-xir", "-o", binary_output, "-"], input=ir, text=True
            )
