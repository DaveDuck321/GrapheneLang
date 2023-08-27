function main : () -> int = {
    let a : StringView = sv(");
}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 2
/// let a : StringView = sv(?);
/// ^
/// Error: invalid syntax, unterminated string literal
