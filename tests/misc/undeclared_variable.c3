function main: () -> int = {
    return x;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2, in 'function main: () -> int'
/// Error: could not find variable definition 'x'
