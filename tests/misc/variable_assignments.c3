@operator + : (lhs : int, rhs: int) -> int = {
    return __builtin_add(lhs, rhs);
}

function main : () -> int = {
    mut x : int = 1;
    mut a : int = 2;
    let b : int mut& = &mut a;

    x = 3;
    b = 4;  // Sets `a` to 4
    return a + x + b;
}

/// @COMPILE
/// @RUN; EXPECT 11
