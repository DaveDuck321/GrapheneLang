function main: () -> int = {
    let x: int;
    return x;
};

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 3, in 'function main: () -> int'
/// Cannot use uninitialized variable 'x'
