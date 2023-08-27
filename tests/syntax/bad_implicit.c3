@implicit hasnext : () -> bool = {}

/// @COMPILE; EXPECT ERR
/// File '*.c3', line 1
/// [@]implicit hasnext : () -> bool = {}
/// ^
/// Error: invalid syntax, only 'has_next', 'get_next', and 'destruct' are valid implicit function names
