function main : () -> int = {
    0x;
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2
/// 0x;
/// ^
/// Error: invalid syntax, expected hex constant continuation
