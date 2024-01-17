function main : () -> int = {
    return;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2, in 'function main: () -> int'
/// Error in return statement: type 'int' does not match expected type 'void'
