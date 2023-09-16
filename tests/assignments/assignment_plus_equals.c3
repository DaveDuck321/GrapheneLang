@assignment += : (lhs : int mut&, rhs: int) -> void = {
    lhs = __builtin_add(lhs, rhs);
}

function main : () -> int = {
    mut a : int = 0;
    a += 7;
    return a;
}

/// @COMPILE
/// @RUN; EXPECT 7
