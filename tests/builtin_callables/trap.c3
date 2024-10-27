function main : () -> int = {
    // lli can't handle the trap instruction...
    // Instead use `opt` to check that the codegen is valid
    __builtin_trap();
    return 0;
}

/// @COMPILE
/// @OPTIMIZE
/// @GREP_IR call void @llvm.trap()
