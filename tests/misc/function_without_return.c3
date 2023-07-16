function without_return : (a : bool) -> int = {
    if (a) {
        return 7;
    }
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 5, in 'function without_return: (bool) -> int'
/// Error: control flow reaches end of non-void function
