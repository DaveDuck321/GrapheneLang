function main: () -> int = {
    let a : int = 120;
    let b : i8 = __builtin_narrow<i8>(a);
    let c : i8 = __builtin_narrow<i8>(7);
    return __builtin_add(b, c);
}

/// @COMPILE
/// @RUN; EXPECT 127
