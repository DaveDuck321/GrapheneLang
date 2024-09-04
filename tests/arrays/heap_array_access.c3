@require_once "std/array.c3"

function main : () -> int = {
    let str : u8[&] = "ABCD";

    return __builtin_bitcast<i8>(str[2]);
}

/// @COMPILE
/// @RUN; EXPECT 67
