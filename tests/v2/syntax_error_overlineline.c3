function main : () -> int = {
    // TODO: test caret position
    return 0
};

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 4
/// return 0
/// };
/// ^
/// Error: invalid syntax, unexpected symbol
