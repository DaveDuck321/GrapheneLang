function main : () -> int = {
    const str : u8[&] = "ABCD";

    return __builtin_bitcast<i8>(str[2]);
};

/// @COMPILE
/// @RUN; EXPECT 67
