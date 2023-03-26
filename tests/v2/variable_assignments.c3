@operator + : (lhs : int, rhs: int) -> int = {
    return __builtin_add(lhs, rhs);
};

function main : () -> int = {
    let x : int = 1;
    let a : int = 2;
    let b : int& = &a;

    x = 3;
    b = 4;  // Sets `a` to 4
    return a + x + b;
};

/// @COMPILE
/// @RUN; EXPECT 11
