typedef MyStruct : {a: int, x: int};

@operator + : (lhs : int, rhs: int) -> int = {
    return __builtin_add(lhs, rhs);
};

function main : () -> int = {
    let struct : MyStruct;
    let b : int& = &struct.a;

    b = 6;  // Sets `struct.a` to 6
    struct.x = 5;  // Sets `struct.x` to 5
    return struct.a + struct.x + 5;
};

/// @COMPILE
/// @RUN; EXPECT 16
