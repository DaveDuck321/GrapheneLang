function main : () -> int = {
    let a : int = 9;
    &a = 10;
    return a;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 3, in 'function main: () -> int'
/// Error: cannot assign to borrowed reference type 'int&'. Please do not manually borrow before an assignment.
