function [T+B] main : () -> int = {}
/// @COMPILE; EXPECT ERR
/// File '*.c3', line 1
/// function [[]T+B[]] main : () -> int = {}
/// ^
/// Error: invalid syntax, expected ',' or ']'
