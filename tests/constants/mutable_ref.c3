function main : () -> int = {
    let a : int = 0;
    mut b : int& = &a;
    return a;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 3, in 'function main: () -> int'
/// Error: cannot store reference type 'int&' in mutable variable 'b', consider using 'let b : int& = ...'
