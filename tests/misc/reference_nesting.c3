typedef int_ref : int&

function d : (var : int_ref&) -> int = { return var; }

function main: () -> int = {
    let a: int = 2;
    let b: int& = &a;
    let c: int_ref& = &b;
    return d(&c);
}

/// @COMPILE; EXPECT ERR
/// File '*/reference_nesting.c3', line 3, in 'function signature'
/// Error: cannot construct reference type since 'typedef int_ref = int&' is already a reference
