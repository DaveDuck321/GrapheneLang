typedef Array : u8[8a7]

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 1
/// typedef Array : u8[[]8a7[]]
/// ^
/// Error: invalid syntax, illegal digit 'a' in decimal constant
