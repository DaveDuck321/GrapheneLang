typedef MyStruct : {a: int, x: int}

function main : () -> int = {
    let struct : MyStruct;
    struct = {9, 10};
    struct = {.x = 11, .a = 12};

    return struct.x;
}

/// @COMPILE
/// @RUN; EXPECT 11
