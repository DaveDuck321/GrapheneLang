typedef S : {
    a: i32,
    a: i64  // We should print the correct line number.
}

function main : () -> int = {
    const s : S = {0, 0};
    return s.a;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 3, in 'struct definition'
/// Error: redeclaration of struct member `a` (previous declaration `a: i32`)
