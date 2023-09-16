function test : (a : int&) -> int& = { return &a; }

function main : () -> int = {
    mut a : int = 1;
    test(&a) = 6;
    return 0;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 5, in 'function main: () -> int'
/// Error: left hand side of assignment operation has non-mutable type 'int&'
