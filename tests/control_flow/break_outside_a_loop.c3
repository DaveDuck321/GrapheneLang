function main : () -> int = {
    break;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2, in 'function main: () -> int'
/// 'break' is not inside a loop
