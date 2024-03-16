function main : () -> int = {
    continue;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2, in 'function main: () -> int'
/// 'continue' is not inside a loop
