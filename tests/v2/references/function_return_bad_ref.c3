function test_fn : (arg : int&) -> int& = {
    return arg;
};

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2, in 'function test_fn: (int&) -> int&'
/// Error in return statement: type 'int' does not match expected type 'int&'
