function test_fn : (arg : int) -> int& = {
    // This should not give an error message that recommends a borrow since arg would be out-of-scope.
    return arg;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 3, in 'function test_fn: (int) -> int&'
/// Error in return statement: type 'int' does not match expected type 'int&'
