function main : () -> int = {
    // We don't use the stdlib helpers here so only one function is emitted
    // and our GREP is less likely to match some internals.

    mut a : int = 0;    // Non-volatile assign
    a = 7;              // Non-volatile assign

    __builtin_volatile_write(&mut a, a);    // Volatile assign, non-volatile read
    a = __builtin_volatile_read(&a);        // Volatile read

    // The optimizer should:
    // 1) Squash the first two writes and the first read into a single constant
    // 2) Preserve the order/ count of volatile operations
    // We test 1) but not 2) (we've emitted the correct ir, lets just trust LLVM here)

    return a;
}

/// @COMPILE
/// @OPTIMIZE --strip-debug
/// @GREP_IR %0 = alloca i32, align 4
/// @GREP_IR store volatile i32 7, ptr %0, align 4
/// @GREP_IR load volatile i32, ptr %0, align 4
/// @RUN; EXPECT 7
