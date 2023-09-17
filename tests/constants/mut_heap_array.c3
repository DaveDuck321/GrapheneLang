function main : () -> int = {
    let a : int[3] = {1, 2, 3};
    mut b : int[mut&] = &a;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 3, in 'function main: () -> int'
/// Error: cannot store reference type 'int[[]mut&[]]' in mutable variable 'b', consider using 'let b : int[[]mut&[]] = ...'
