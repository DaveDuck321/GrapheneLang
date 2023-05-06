function test_fn : (arg : int&) -> int& = {
    return &arg;
};

function main : () -> int = {
    let a : int = 0;
    let a_ref : int& = test_fn(&a);

    a_ref = 10;

    return a;
};

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 7, in 'function main: () -> int'
/// Error in variable assignment: type 'int' does not match expected type 'int&'
/// Type 'int' is implicitly dereferenced here, did you mean to borrow using '&...'?
